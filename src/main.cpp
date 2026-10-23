#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace geode::prelude;

// ---- the network: 10 inputs, three hidden layers, one output (jump or not)
static constexpr int NL = 5;
static constexpr int LAYERS[NL] = {10, 16, 12, 8, 1};
static constexpr int POP = 30;
static constexpr int ELITE = 5;

static size_t weightCount() {
    size_t n = 0;
    for (int l = 0; l + 1 < NL; l++) n += (size_t)(LAYERS[l] + 1) * LAYERS[l + 1];
    return n;
}

struct Genome {
    std::vector<float> w;
    float fit = 0.f;
};

static std::mt19937& rng() {
    static std::mt19937 r{std::random_device{}()};
    return r;
}
static float gauss(float s) { return std::normal_distribution<float>(0.f, s)(rng()); }
static float uni() { return std::uniform_real_distribution<float>(0.f, 1.f)(rng()); }

// one pass through the network; acts[l] holds what every node of layer l did
static void forward(const Genome& g, const float* in, std::vector<float> acts[NL]) {
    acts[0].assign(in, in + LAYERS[0]);
    size_t p = 0;
    for (int l = 0; l + 1 < NL; l++) {
        acts[l + 1].assign(LAYERS[l + 1], 0.f);
        for (int o = 0; o < LAYERS[l + 1]; o++) {
            float s = g.w[p + LAYERS[l]]; // bias
            for (int i = 0; i < LAYERS[l]; i++) s += g.w[p + i] * acts[l][i];
            p += LAYERS[l] + 1;
            acts[l + 1][o] = std::tanh(s);
        }
    }
}

// ---- the evolution (lives as long as the game is open, so it keeps learning between attempts)
struct Evo {
    std::vector<Genome> pop;
    int cur = 0, gen = 1;
    float bestEver = 0.f, bestGen = 0.f;

    void init() {
        if (!pop.empty()) return;
        pop.resize(POP);
        for (auto& g : pop) {
            g.w.resize(weightCount());
            for (auto& x : g.w) x = gauss(0.8f);
        }
    }
    void finish(float fit) {
        pop[cur].fit = fit;
        bestGen = std::max(bestGen, fit);
        bestEver = std::max(bestEver, fit);
        if (++cur >= POP) { evolve(); cur = 0; gen++; bestGen = 0.f; }
    }
    void evolve() {
        std::sort(pop.begin(), pop.end(), [](const Genome& a, const Genome& b) { return a.fit > b.fit; });
        std::vector<Genome> next(pop.begin(), pop.begin() + ELITE);
        while ((int)next.size() < POP) {
            const Genome& a = pop[rng()() % ELITE];
            const Genome& b = pop[rng()() % ELITE];
            Genome c;
            c.w.resize(a.w.size());
            for (size_t i = 0; i < c.w.size(); i++) {
                c.w[i] = uni() < 0.5f ? a.w[i] : b.w[i];
                if (uni() < 0.12f) c.w[i] += gauss(0.35f);
            }
            next.push_back(std::move(c));
        }
        for (auto& g : next) g.fit = 0.f;
        pop = std::move(next);
    }
};
static Evo g_evo;

struct Box { float l, r, y; };

class $modify(NeuroLayer, PlayLayer) {
    struct Fields {
        std::vector<Box> hazards, solids; // sorted by their right edge
        CCDrawNode* draw = nullptr;
        CCLabelBMFont* label = nullptr;
        bool dead = false, holding = false;
        int frame = 0;
        float scale = 1.f;
    };

    bool on() { return Mod::get()->getSettingValue<bool>("enabled"); }

    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();
        auto f = m_fields.self();
        for (auto obj : CCArrayExt<GameObject*>(m_objects)) {
            if (!obj) continue;
            auto t = obj->m_objectType;
            if (t != GameObjectType::Hazard && t != GameObjectType::Solid) continue;
            auto r = obj->getObjectRect();
            Box b{r.getMinX(), r.getMaxX(), r.getMidY()};
            (t == GameObjectType::Hazard ? f->hazards : f->solids).push_back(b);
        }
        auto byRight = [](const Box& a, const Box& b) { return a.r < b.r; };
        std::sort(f->hazards.begin(), f->hazards.end(), byRight);
        std::sort(f->solids.begin(), f->solids.end(), byRight);

        g_evo.init();
        auto size = CCDirector::get()->getWinSize();
        f->draw = CCDrawNode::create();
        f->draw->setZOrder(1000);
        m_uiLayer->addChild(f->draw);
        f->label = CCLabelBMFont::create("", "chatFont.fnt");
        f->label->setAnchorPoint({0.f, 1.f});
        f->label->setScale(0.28f);
        f->label->setPosition({4.f, size.height - 3.f});
        f->label->setZOrder(1001);
        m_uiLayer->addChild(f->label);
        applySpeed();
    }

    void applySpeed() {
        float s = on() ? (float)Mod::get()->getSettingValue<double>("speed") : 1.f;
        CCDirector::get()->getScheduler()->setTimeScale(s);
    }

    void onQuit() {
        CCDirector::get()->getScheduler()->setTimeScale(1.f);
        PlayLayer::onQuit();
    }

    void destroyPlayer(PlayerObject* p, GameObject* obj) {
        PlayLayer::destroyPlayer(p, obj);
        auto f = m_fields.self();
        if (!on() || p != m_player1 || f->dead) return;
        f->dead = true;
        g_evo.finish(m_player1->getPositionX());
    }

    // the nearest things ahead, as numbers the network can use
    void sense(float* in) {
        auto f = m_fields.self();
        float px = m_player1->getPositionX(), py = m_player1->getPositionY();
        in[0] = std::clamp((float)m_player1->m_yVelocity / 20.f, -2.f, 2.f);
        in[1] = m_player1->m_isOnGround ? 1.f : 0.f;
        auto fill = [&](const std::vector<Box>& v, int count, float* out) {
            auto it = std::lower_bound(v.begin(), v.end(), px - 10.f, [](const Box& b, float x) { return b.r < x; });
            for (int k = 0; k < count; k++, ++it) {
                if (it == v.end()) { out[2 * k] = 4.f; out[2 * k + 1] = 0.f; continue; }
                out[2 * k] = std::clamp((it->l - px) / 150.f, -0.5f, 4.f);
                out[2 * k + 1] = std::clamp((it->y - py) / 60.f, -3.f, 3.f);
            }
        };
        fill(f->hazards, 3, in + 2);
        fill(f->solids, 1, in + 8);
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        auto f = m_fields.self();
        if (!f->draw) return;
        bool active = on();
        f->draw->setVisible(active && Mod::get()->getSettingValue<bool>("show-network"));
        f->label->setVisible(active);
        if (!active) return;

        if (f->dead) { // the attempt is over: start the next network straight away
            f->dead = false;
            f->holding = false;
            resetLevel();
            return;
        }
        if (m_player1->m_isDead) return;

        float in[10];
        sense(in);
        std::vector<float> acts[NL];
        forward(g_evo.pop[g_evo.cur], in, acts);
        bool jump = acts[NL - 1][0] > 0.f;
        if (jump && !f->holding) { handleButton(true, 1, true); f->holding = true; }
        else if (!jump && f->holding) { handleButton(false, 1, true); f->holding = false; }

        f->label->setString(fmt::format("NeuroBot  generation {}  network {}/{}  best this generation {:.0f}  best ever {:.0f}",
            g_evo.gen, g_evo.cur + 1, POP, g_evo.bestGen, g_evo.bestEver).c_str());
        if (f->draw->isVisible() && (++f->frame % 2) == 0) drawNet(acts);
    }

    // the picture of the network: lines are the weights (green positive, red negative), dots are the nodes
    void drawNet(std::vector<float> acts[NL]) {
        auto f = m_fields.self();
        auto size = CCDirector::get()->getWinSize();
        const float W = 150.f, H = 95.f, x0 = 4.f, y0 = size.height - 14.f - H;
        f->draw->clear();
        CCPoint box[4] = {{x0, y0}, {x0 + W, y0}, {x0 + W, y0 + H}, {x0, y0 + H}};
        f->draw->drawPolygon(box, 4, ccc4f(0.f, 0.f, 0.f, 0.55f), 0.5f, ccc4f(1.f, 1.f, 1.f, 0.25f));

        auto pos = [&](int l, int i) {
            float x = x0 + 11.f + (W - 22.f) * l / (NL - 1);
            float step = (H - 12.f) / std::max(1, LAYERS[l]);
            float y = y0 + H / 2.f + ((LAYERS[l] - 1) / 2.f - i) * std::min(step, 8.f);
            return CCPoint{x, y};
        };
        const auto& g = g_evo.pop[g_evo.cur];
        size_t p = 0;
        for (int l = 0; l + 1 < NL; l++) {
            for (int o = 0; o < LAYERS[l + 1]; o++) {
                for (int i = 0; i < LAYERS[l]; i++) {
                    float w = g.w[p + i], a = std::abs(w * acts[l][i]);
                    float al = std::clamp(0.06f + a * 0.5f, 0.f, 0.9f);
                    f->draw->drawSegment(pos(l, i), pos(l + 1, o), 0.2f + std::min(a, 1.5f) * 0.25f,
                        w >= 0 ? ccc4f(0.3f, 0.9f, 0.5f, al) : ccc4f(1.f, 0.35f, 0.3f, al));
                }
                p += LAYERS[l] + 1;
            }
        }
        for (int l = 0; l < NL; l++)
            for (int i = 0; i < LAYERS[l]; i++) {
                float a = std::clamp(acts[l][i], -1.f, 1.f);
                float v = (a + 1.f) / 2.f; // dark = -1, bright = +1
                bool out = l == NL - 1;
                f->draw->drawDot(pos(l, i), out ? 3.5f : 2.2f,
                    out ? ccc4f(v, v * 0.8f + 0.2f, 0.2f, 1.f) : ccc4f(0.15f + v * 0.85f, 0.2f + v * 0.7f, 0.35f + v * 0.65f, 1.f));
            }
    }
};
