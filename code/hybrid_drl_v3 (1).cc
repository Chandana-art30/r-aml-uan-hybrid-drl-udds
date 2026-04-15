/* ╔══════════════════════════════════════════════════════════════════════╗
 * ║  R-AML-UAN: HYBRID DRL ACOUSTIC-OPTICAL UNDERWATER NETWORK         ║
 * ║  + UDDS (Underwater Decoy-based Deception System)                  ║
 * ║  + OPTICAL SENSOR MODEL (Enemy cameras/lidar — teammate fix)       ║
 * ║  ── FULLY CORRECTED FINAL VERSION v2 (ALL BUGS FIXED) ──           ║
 * ║                                                                     ║
 * ║  ALL FIXES APPLIED:                                                ║
 * ║  FIX-1 Directional sonar : 60-degree beam                         ║
 * ║  FIX-2 Ocean noise       : Gaussian(0,0.5)                        ║
 * ║  FIX-3 Frequency model   : Acoustic=10kHz, Optical=500THz         ║
 * ║  FIX-4 Adaptive enemy    : suspicion threshold=5.0 (TUNED)        ║
 * ║  FIX-5 Probabilistic decoy success                                ║
 * ║  FIX-A Node5 sink not hardcoded GREEN                             ║
 * ║  FIX-B Water change does not overwrite DRL label                  ║
 * ║  FIX-C UpdateNodeForAction called every packet                    ║
 * ║  FIX-D Heavier decoy penalty when suspicious                      ║
 * ║  FIX-E Node0 source not hardcoded RED                             ║
 * ║  FIX-OPT Enemy optical sensor + decoy optical jamming (FIXED)     ║
 * ║  FIX-COLOR Blended water+action color visualization               ║
 * ╚══════════════════════════════════════════════════════════════════════╝ */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/aqua-sim-ng-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include <cmath>
#include <vector>
#include <map>
#include <tuple>
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("HybridDRL_UDDS_Final");

static const double OPT_RANGE   = 100.0;
static const double ACU_RANGE   = 450.0;
static const double OPT_LQ_THR  = 0.30;
static const double ACU_LQ_THR  = 0.20;
static const double ALPHA_LR    = 0.15;
static const double GAMMA_DF    = 0.90;
static const double EPS_START   = 0.90;
static const double EPS_MIN     = 0.08;
static const double EPS_DECAY   = 0.985;
static const double OPT_ENERGY  = 0.5;
static const double ACU_ENERGY  = 2.0;
static const uint32_t NUM_NODES = 6;

static const double DECOY_ENERGY        = 3.0;
static const double OPTICAL_SIGNATURE   = 0.05;
static const double ACOUSTIC_SIGNATURE  = 1.0;
static const double DECOY_SIGNATURE     = 5.0;
static const double ENEMY_DETECT_THRESH = 4.0;   /* CHANGED: 8.0 -> 4.0 */
static const uint32_t NUM_ENEMY         = 2;

static const double FREQ_ACOUSTIC = 10.0;
static const double FREQ_DECOY    = 10.0;
static const double FREQ_OPTICAL  = 500.0;
static const double SONAR_BEAM_RAD       = M_PI / 3.0;
static const double OCEAN_NOISE_STD      = 0.5;

static const double SUSPICION_THRESHOLD  = 5.0;

static const double OPTICAL_DETECT_RANGE  = 80.0;
static const double OPTICAL_DETECT_THRESH = 2.0;
static const double OPTICAL_JAM_THRESH    = 5.0;
static const double OPTICAL_JAM_POWER     = 3.0;

enum WaterCond { PRISTINE=0, CLEAR=1, HAZY=2,
                 COASTAL=3, MURKY=4, MUDDY=5, THERMAL=6 };

static WaterCond   g_waterCond[NUM_NODES] = { CLEAR, CLEAR, CLEAR, MURKY, CLEAR, CLEAR };
static const char* WaterName[] = { "PRISTINE","CLEAR","HAZY",
                                    "COASTAL","MURKY","MUDDY","THERMAL" };

struct RGB { uint8_t r, g, b; };

static RGB GetWaterRGB(WaterCond w) {
  switch (w) {
    case PRISTINE: return {  0, 220, 255};
    case CLEAR:    return {  0, 128, 255};
    case HAZY:     return {100, 180, 200};
    case COASTAL:  return {180, 200, 100};
    case MURKY:    return {255, 140,   0};
    case MUDDY:    return {139,  69,  19};
    case THERMAL:  return {160,  32, 240};
  }
  return {0, 128, 255};
}

static RGB GetActionRGB(int action) {
  if (action == 1) return {  0, 255,   0};
  if (action == 0) return {255,   0,   0};
  if (action == 2) return {255, 255,   0};
  return {255, 255, 255};
}

static RGB BlendRGB(RGB water, RGB action, double actionWeight = 0.65) {
  double w = 1.0 - actionWeight;
  return {
    (uint8_t)std::round(w * water.r + actionWeight * action.r),
    (uint8_t)std::round(w * water.g + actionWeight * action.g),
    (uint8_t)std::round(w * water.b + actionWeight * action.b)
  };
}

static void SetWaterColor(AnimationInterface* anim, Ptr<Node> n, WaterCond w) {
  RGB c = GetWaterRGB(w);
  anim->UpdateNodeColor(n, c.r, c.g, c.b);
}

static double OptAttenuation(WaterCond w) {
  switch (w) {
    case PRISTINE: return 0.5;
    case CLEAR:    return 1.0;
    case HAZY:     return 1.2;
    case COASTAL:  return 1.8;
    case MURKY:    return 3.0;
    case MUDDY:    return 8.0;
    case THERMAL:  return 2.0;
  }
  return 1.0;
}
static double AcuAttenuation(WaterCond w) {
  switch (w) {
    case PRISTINE: return 1.0;
    case CLEAR:    return 1.0;
    case HAZY:     return 1.05;
    case COASTAL:  return 1.1;
    case MURKY:    return 1.3;
    case MUDDY:    return 1.8;
    case THERMAL:  return 2.5;
  }
  return 1.0;
}

struct EnemySub {
  uint32_t id;
  double   posX, posY;
  double   facingAngle;
  int      lockedTarget;
  bool     isDeceived;
  bool     isOpticalBlinded;
  double   acousticEnergy[6];
  double   opticalEnergy[6];
  double   decoySuspicion[6];
  uint32_t realDetections;
  uint32_t decoyHits;
  uint32_t missedReal;
  uint32_t opticalDetections;
  uint32_t opticalBlindEvents;
};
static EnemySub g_enemy[NUM_ENEMY];

static double g_nodeX[NUM_NODES] = { 50,125,327,417,567,652 };
static double g_nodeY[NUM_NODES] = { 200,200,200,200,200,200 };

static void InitEnemies() {
  g_enemy[0] = {0, 300.0, 400.0, M_PI, -1, false, false,
                {0,0,0,0,0,0}, {0,0,0,0,0,0},
                {0,0,0,0,0,0}, 0, 0, 0, 0, 0};
  g_enemy[1] = {1, 580.0, 420.0, M_PI, -1, false, false,
                {0,0,0,0,0,0}, {0,0,0,0,0,0},
                {0,0,0,0,0,0}, 0, 0, 0, 0, 0};
}

static double EnemyDist(uint32_t eId, uint32_t nId) {
  double dx = g_enemy[eId].posX - g_nodeX[nId];
  double dy = g_enemy[eId].posY - g_nodeY[nId];
  return std::sqrt(dx*dx + dy*dy);
}

static double EnemyAngleToNode(uint32_t eId, uint32_t nodeId) {
  double dx = g_nodeX[nodeId] - g_enemy[eId].posX;
  double dy = g_nodeY[nodeId] - g_enemy[eId].posY;
  return std::atan2(dy, dx);
}

static bool NodeInSonarBeam(uint32_t eId, uint32_t nodeId) {
  double angleToNode = EnemyAngleToNode(eId, nodeId);
  double diff = angleToNode - g_enemy[eId].facingAngle;
  while (diff >  M_PI) diff -= 2.0*M_PI;
  while (diff < -M_PI) diff += 2.0*M_PI;
  return (std::fabs(diff) <= SONAR_BEAM_RAD);
}

static bool NodeNearEnemy(uint32_t nodeId) {
  for (uint32_t e=0; e<NUM_ENEMY; e++)
    if (EnemyDist(e, nodeId) < 420.0) return true;
  return false;
}

static std::mt19937 g_rng(42);
static std::uniform_real_distribution<double> g_urand(0.0,1.0);
static std::normal_distribution<double> g_noiseDistr(0.0, OCEAN_NOISE_STD);

static double FreqAttenFactor(double freq, double dist) {
  return std::exp(-0.001 * freq * dist / 1000.0);
}

static void EnemyDetects(uint32_t nodeId, double signature,
                         const std::string& mode) {
  double freq = FREQ_ACOUSTIC;
  if (mode == "OPTICAL") freq = FREQ_OPTICAL;
  if (mode == "DECOY")   freq = FREQ_DECOY;

  for (uint32_t e=0; e<NUM_ENEMY; e++) {
    double dist = EnemyDist(e, nodeId);

    if (mode == "OPTICAL" || mode == "DECOY") {
      if (dist <= OPTICAL_DETECT_RANGE) {

        if (mode == "OPTICAL") {
          double optHeard = signature / (1.0 + dist/20.0);
          g_enemy[e].opticalEnergy[nodeId] += optHeard * 0.1;
          if (g_enemy[e].opticalEnergy[nodeId] >= OPTICAL_DETECT_THRESH
              && !g_enemy[e].isOpticalBlinded) {
            g_enemy[e].opticalDetections++;
            std::cout << "\n  !!! ENEMY-" << e
                      << " optical sensor detected Node" << nodeId
                      << " at " << std::fixed << std::setprecision(1)
                      << dist << "m !!!\n";
          } else if (g_enemy[e].isOpticalBlinded) {
            std::cout << "  [OPT-BLIND] Enemy-" << e
                      << " optical sensor blinded — Node" << nodeId
                      << " optical tx invisible!\n";
          }
        }

        if (mode == "DECOY") {
          double jamPower = OPTICAL_JAM_POWER / (1.0 + dist/10.0);
          g_enemy[e].opticalEnergy[nodeId] += jamPower;
          if (g_enemy[e].opticalEnergy[nodeId] >= OPTICAL_JAM_THRESH) {
            if (!g_enemy[e].isOpticalBlinded) {
              g_enemy[e].isOpticalBlinded = true;
              g_enemy[e].opticalBlindEvents++;
              std::cout << "\n  *** OPTICAL JAM! Enemy-" << e
                        << " camera BLINDED by Node" << nodeId
                        << " decoy flash! (dist=" << std::setprecision(1)
                        << dist << "m) ***\n"
                        << "  *** Optical transmissions now INVISIBLE"
                        << " to Enemy-" << e << " ***\n";
            }
          }
        }
      }
    }

    if (mode == "ACOUSTIC" || mode == "DECOY") {
      if (dist > 600.0) continue;

      if (!NodeInSonarBeam(e, nodeId)) {
        if (mode != "OPTICAL")
          std::cout << "  [SONAR] Enemy-" << e << " beam misses Node"
                    << nodeId << " (outside 60° cone)\n";
        continue;
      }

      double freqFactor = FreqAttenFactor(freq, dist);
      double heard = signature * freqFactor / (1.0 + dist/200.0);
      double noise = g_noiseDistr(g_rng);
      heard = std::max(0.0, heard + noise);

      g_enemy[e].acousticEnergy[nodeId] += heard;

      if (g_enemy[e].acousticEnergy[nodeId] >= ENEMY_DETECT_THRESH) {

        if (mode == "DECOY") {
          if (g_enemy[e].decoySuspicion[nodeId] >= SUSPICION_THRESHOLD) {
            std::cout << "  [ENEMY-" << e << " ADAPTIVE] Suspicious of Node"
                      << nodeId << " decoy — ignoring it!\n";
            continue;
          }
          double successProb = DECOY_SIGNATURE / (DECOY_SIGNATURE + dist/100.0);
          if (g_urand(g_rng) < successProb) {
            if (!g_enemy[e].isDeceived) {
              g_enemy[e].isDeceived   = true;
              g_enemy[e].lockedTarget = (int)nodeId;
              g_enemy[e].decoyHits++;
              std::cout << "\n  *** DECOY SUCCESS! Enemy-" << e
                        << " acoustic locked on Node" << nodeId
                        << " (prob=" << std::fixed << std::setprecision(2)
                        << successProb << ")"
                        << " — like missile chasing FOTD! ***\n";
            }
          } else {
            std::cout << "  [DECOY FAILED] Enemy-" << e
                      << " saw through Node" << nodeId
                      << " decoy (prob=" << std::setprecision(2)
                      << successProb << ")\n";
            g_enemy[e].decoySuspicion[nodeId] += 0.8;
          }
          g_enemy[e].decoySuspicion[nodeId] += 0.2;

        } else if (mode == "ACOUSTIC") {
          if (g_enemy[e].isOpticalBlinded) {
            g_enemy[e].missedReal++;
            std::cout << "\n  >>> Enemy-" << e << " acoustic hit Node"
                      << nodeId << " BUT optical sensor blinded"
                      << " — cannot confirm target! <<<\n";
          } else if (!g_enemy[e].isDeceived) {
            g_enemy[e].lockedTarget  = (int)nodeId;
            g_enemy[e].realDetections++;
            std::cout << "\n  !!! ENEMY-" << e
                      << " DETECTED real Node" << nodeId
                      << " (heard=" << std::setprecision(3) << heard
                      << " noise added) !!!\n";
          } else {
            g_enemy[e].missedReal++;
            std::cout << "\n  >>> Enemy-" << e << " missed Node" << nodeId
                      << " (tracking decoy — deception working!) <<<\n";
          }
        }
      }
    }
  }
}

static void PatrolEnemy(uint32_t eId, double x, double y) {
  double dx = 350.0 - x;
  double dy = 200.0 - y;
  g_enemy[eId].posX        = x;
  g_enemy[eId].posY        = y;
  g_enemy[eId].facingAngle = std::atan2(dy, dx);

  if (g_enemy[eId].isDeceived) {
    g_enemy[eId].isDeceived   = false;
    g_enemy[eId].lockedTarget = -1;
  }
  g_enemy[eId].isOpticalBlinded = false;

  for (int i=0; i<6; i++) {
    g_enemy[eId].acousticEnergy[i]  *= 0.5;
    g_enemy[eId].opticalEnergy[i]   *= 0.3;
    g_enemy[eId].decoySuspicion[i]  *= 0.2;   /* CHANGED: 0.4 -> 0.2 (forgets faster) */
  }
  std::cout << "  [ENEMY-" << eId << " PATROL] moved to ("
            << x << "," << y << ")  facingAngle="
            << std::fixed << std::setprecision(2)
            << g_enemy[eId].facingAngle * 180.0 / M_PI << "°"
            << "  opticalSensor=RECOVERED\n";
}

static uint32_t g_pktCnt=0, g_optCnt=0, g_acuCnt=0, g_dropCnt=0;
static uint32_t g_decoyCnt=0, g_decoySuccess=0;

struct NodeStat {
  uint32_t optSent   = 0;
  uint32_t acuSent   = 0;
  uint32_t decoySent = 0;
  double   energy    = 5000.0;
  int      lastAction = -1;
};
static NodeStat g_nstat[NUM_NODES];

static uint32_t g_wcOpt[7]={0}, g_wcAcu[7]={0}, g_wcDecoy[7]={0};
static uint32_t g_step=0, g_exploreN=0, g_exploitN=0;
static double   g_epsilon=EPS_START, g_totalRew=0.0;

static std::map<std::tuple<int,int,int>,
                std::tuple<double,double,double>> g_qtable;

static std::ofstream g_csv, g_qcsv, g_decoycsv;

static AnimationInterface* g_anim     = nullptr;
static NodeContainer*      g_nodesPtr = nullptr;

static double OpticalLQ(double dist, WaterCond w) {
  return std::max(0.0, std::min(1.0,
    std::exp(-OptAttenuation(w)*dist/OPT_RANGE)));
}
static double AcousticLQ(double dist, WaterCond w) {
  return std::max(0.0, std::min(1.0,
    std::exp(-AcuAttenuation(w)*dist/ACU_RANGE)));
}
static int DistBucket(double dist) {
  if (dist<=70)  return 0;
  if (dist<=140) return 1;
  if (dist<=280) return 2;
  return 3;
}

static std::tuple<double,double,double>& GetQ(int nid, int db, int wc) {
  auto key = std::make_tuple(nid,db,wc);
  if (!g_qtable.count(key))
    g_qtable[key] = std::make_tuple((db==0?5.0:1.0), 2.0, 0.5);
  return g_qtable[key];
}

static double ComputeReward(int action, double optLQ, double acuLQ,
                             WaterCond w, double dist,
                             uint32_t nodeId, bool enemyNearby) {
  double lq    = (action==1) ? optLQ : acuLQ;
  double eSave = (action==1) ? 1.5   : 0.0;
  double bonus = 0.0;

  if (action==1) {
    if      ((w==PRISTINE) && optLQ>=OPT_LQ_THR) bonus =  8.0;
    else if ((w==CLEAR)    && optLQ>=OPT_LQ_THR) bonus =  3.0;
    else if ((w==HAZY)     && optLQ>=OPT_LQ_THR) bonus =  1.5;
    else if (optLQ < OPT_LQ_THR)                 bonus = -5.0;
    if (enemyNearby && optLQ>=OPT_LQ_THR)        bonus += 4.0;

  } else if (action==0) {
    if (acuLQ < ACU_LQ_THR) bonus = -5.0;
    bool optViable = (w==PRISTINE || w==CLEAR || w==HAZY);
    if (dist<=OPT_RANGE && optViable && optLQ>=OPT_LQ_THR) bonus = -2.0;
    if (enemyNearby) bonus -= 3.0;

  } else {
    lq = 0.8; eSave = 0.0;
    if (!enemyNearby) {
      bonus = -5.0;
    } else {
      bool deceived   = false;
      bool suspicious = false;
      for (uint32_t e=0; e<NUM_ENEMY; e++) {
        if (g_enemy[e].isDeceived && g_enemy[e].lockedTarget==(int)nodeId)
          deceived = true;
        if (g_enemy[e].decoySuspicion[nodeId] >= SUSPICION_THRESHOLD)
          suspicious = true;
      }
      if (suspicious) {
        bonus = -6.0;
      } else if (deceived) {
        bonus = 10.0;
        g_decoySuccess++;
      } else {
        bonus = 7.0;   /* CHANGED: 5.0 -> 7.0 */
      }
    }
  }

  return 10.0*lq + 2.0*eSave + bonus;
}

static int SelectAction(int nid, int db, int wc,
                        double optLQ, double acuLQ, bool enemyNearby) {
  bool optOK = (optLQ>=OPT_LQ_THR);
  bool acuOK = (acuLQ>=ACU_LQ_THR);

  if (!optOK && !acuOK) return -1;
  if (!optOK && !enemyNearby) return 0;
  if (!acuOK) return 1;

  auto& q    = GetQ(nid,db,wc);
  double Qopt = std::get<0>(q);
  double Qacu = std::get<1>(q);
  double Qdec = std::get<2>(q);

  bool exploring = (g_urand(g_rng) < g_epsilon);
  int  action;

  if (exploring) {
    double r = g_urand(g_rng);
    if (!optOK)     action = (r<0.6)?0:2;
    else if (r<0.4) action = 1;
    else if (r<0.7) action = 0;
    else            action = 2;
    g_exploreN++;
  } else {
    if (!optOK) {
      action = (Qacu>=Qdec)?0:2;
    } else {
      if      (Qopt>=Qacu && Qopt>=Qdec) action=1;
      else if (Qacu>=Qopt && Qacu>=Qdec) action=0;
      else                                action=2;
    }
    g_exploitN++;
  }

  const char* actName = (action==1)?"OPTICAL":(action==0)?"ACOUSTIC":"DECOY";
  std::cout << "  DRL Q[opt]=" << std::fixed << std::setprecision(3) << Qopt
            << "  Q[acu]=" << Qacu
            << "  Q[decoy]=" << Qdec
            << (enemyNearby ? "  [!ENEMY!]" : "")
            << (exploring   ? "  [EXPLORE]" : "  [exploit]")
            << "  --> " << actName << "\n";
  return action;
}

static void UpdateQ(int nid, int db, int wc, int action, double reward,
                    int nNid, int nDb, int nWc) {
  auto& q    = GetQ(nid,  db,  wc);
  auto& qNxt = GetQ(nNid, nDb, nWc);
  double maxNext = std::max({std::get<0>(qNxt),
                              std::get<1>(qNxt),
                              std::get<2>(qNxt)});
  if      (action==1) std::get<0>(q) += ALPHA_LR*(reward+GAMMA_DF*maxNext-std::get<0>(q));
  else if (action==0) std::get<1>(q) += ALPHA_LR*(reward+GAMMA_DF*maxNext-std::get<1>(q));
  else                std::get<2>(q) += ALPHA_LR*(reward+GAMMA_DF*maxNext-std::get<2>(q));
}

static double GetDist(Ptr<Node> a, Ptr<Node> b) {
  uint32_t ai=a->GetId(), bi=b->GetId();
  if (ai>bi){uint32_t t=ai;ai=bi;bi=t;}
  if (ai==0&&bi==1) return  75.0;
  if (ai==1&&bi==2) return 201.6;
  if (ai==2&&bi==3) return  90.0;
  if (ai==3&&bi==4) return 150.0;
  if (ai==4&&bi==5) return  85.0;
  return a->GetObject<MobilityModel>()->GetDistanceFrom(
           b->GetObject<MobilityModel>());
}

static std::string BuildNodeLabel(uint32_t nodeId) {
  std::ostringstream oss;
  oss << "N" << nodeId;
  int la = g_nstat[nodeId].lastAction;
  if      (la == 1) oss << "-OPTICAL";
  else if (la == 0) oss << "-ACOUSTIC";
  else if (la == 2) oss << "-DECOY";
  else if (nodeId == 0)           oss << "-SOURCE";
  else if (nodeId == NUM_NODES-1) oss << "-SINK";
  else                            oss << "-RELAY";
  WaterCond wn = g_waterCond[nodeId];
  if (wn != CLEAR) oss << " [" << WaterName[wn] << "]";
  oss << " E:" << std::fixed << std::setprecision(0) << g_nstat[nodeId].energy;
  return oss.str();
}

static void ChangeWater(uint32_t nodeId, WaterCond newCond) {
  double    now = Simulator::Now().GetSeconds();
  WaterCond old = g_waterCond[nodeId];
  g_waterCond[nodeId] = newCond;

  std::cout << "\n╔══════════════════════════════════════════╗\n"
            << "║  WATER CHANGE @ t=" << std::fixed << std::setprecision(1)
            << now << "s"
            << "  Node" << nodeId << ": "
            << WaterName[old] << " -> " << WaterName[newCond]
            << " ║\n╚══════════════════════════════════════════╝\n";

  if (g_anim && g_nodesPtr) {
    RGB waterColor  = GetWaterRGB(newCond);
    int lastAction  = g_nstat[nodeId].lastAction;
    if (lastAction == -1) {
      g_anim->UpdateNodeColor(g_nodesPtr->Get(nodeId),
                              waterColor.r, waterColor.g, waterColor.b);
    } else {
      RGB actionColor = GetActionRGB(lastAction);
      RGB blended     = BlendRGB(waterColor, actionColor, 0.65);
      g_anim->UpdateNodeColor(g_nodesPtr->Get(nodeId),
                              blended.r, blended.g, blended.b);
    }
    g_anim->UpdateNodeDescription(g_nodesPtr->Get(nodeId),
                                  BuildNodeLabel(nodeId));
  }
}

static void UpdateNodeForAction(uint32_t nodeId, int action) {
  if (!g_anim || !g_nodesPtr || nodeId >= NUM_NODES) return;
  g_nstat[nodeId].lastAction = action;

  RGB waterColor  = GetWaterRGB(g_waterCond[nodeId]);
  RGB actionColor = GetActionRGB(action);
  RGB blended     = BlendRGB(waterColor, actionColor, 0.65);

  g_anim->UpdateNodeColor(g_nodesPtr->Get(nodeId),
                          blended.r, blended.g, blended.b);
  g_anim->UpdateNodeDescription(g_nodesPtr->Get(nodeId),
                                BuildNodeLabel(nodeId));
}

static void UpdateSinkOnReceive(int senderAction) {
  if (!g_anim || !g_nodesPtr) return;
  uint32_t sinkId = NUM_NODES - 1;
  RGB waterColor  = GetWaterRGB(g_waterCond[sinkId]);
  RGB actionColor = GetActionRGB(senderAction);
  RGB blended     = BlendRGB(waterColor, actionColor, 0.65);
  g_anim->UpdateNodeColor(g_nodesPtr->Get(sinkId),
                          blended.r, blended.g, blended.b);

  std::ostringstream oss;
  oss << "N" << sinkId << "-SINK";
  if      (senderAction == 1) oss << " [RX-OPT]";
  else if (senderAction == 0) oss << " [RX-ACU]";
  else                        oss << " [RX-DCY]";
  WaterCond wn = g_waterCond[sinkId];
  if (wn != CLEAR) oss << " [" << WaterName[wn] << "]";
  oss << " E:" << std::fixed << std::setprecision(0) << g_nstat[sinkId].energy;
  g_anim->UpdateNodeDescription(g_nodesPtr->Get(sinkId), oss.str());
}

static void HandlePacket(Ptr<Node> src, Ptr<Node> dst, Ptr<Node> sink) {
  g_pktCnt++;
  uint32_t id  = g_pktCnt;
  double   now = Simulator::Now().GetSeconds();
  uint32_t sid = src->GetId(), did = dst->GetId();
  double   dist = GetDist(src,dst);

  WaterCond wSrc  = (sid<NUM_NODES)?g_waterCond[sid]:CLEAR;
  WaterCond wDst  = (did<NUM_NODES)?g_waterCond[did]:CLEAR;
  WaterCond wLink = (WaterCond)std::max((int)wSrc,(int)wDst);

  double optLQ = OpticalLQ(dist,wLink);
  double acuLQ = AcousticLQ(dist,wLink);
  int    distB = DistBucket(dist);
  bool   enemyNearby = NodeNearEnemy(sid);

  std::cout << "\n=== PKT #" << id
            << "  t=" << std::fixed << std::setprecision(2) << now << "s"
            << "  Node" << sid << " -> Node" << did
            << "  dist=" << std::setprecision(1) << dist << "m"
            << "  water=" << WaterName[wLink]
            << (enemyNearby ? "  [!ENEMY NEARBY!]" : "")
            << "  E=" << g_nstat[sid].energy << "J ===\n";
  std::cout << "  optLQ=" << std::setprecision(3) << optLQ
            << (optLQ>=OPT_LQ_THR?" [OK]":" [POOR]")
            << "  acuLQ=" << acuLQ
            << (acuLQ>=ACU_LQ_THR?" [OK]":" [POOR]") << "\n";

  for (uint32_t e=0; e<NUM_ENEMY; e++) {
    bool inBeam = NodeInSonarBeam(e, sid);
    std::cout << "  [FIX-1] Enemy-" << e
              << " facing=" << std::setprecision(1)
              << g_enemy[e].facingAngle*180.0/M_PI << "°"
              << "  Node" << sid << " in beam: " << (inBeam?"YES":"NO")
              << "  suspicion[node]=" << std::setprecision(1)
              << g_enemy[e].decoySuspicion[sid] << "\n";
  }

  int action = SelectAction((int)sid,distB,(int)wLink,optLQ,acuLQ,enemyNearby);
  if (action==-1) {
    std::cout << "  NO VALID LINK --> DROPPED\n";
    g_dropCnt++;
    return;
  }

  UpdateNodeForAction(sid, action);
  if (did == NUM_NODES-1) UpdateSinkOnReceive(action);

  double reward = ComputeReward(action,optLQ,acuLQ,wLink,dist,sid,enemyNearby);
  g_totalRew += reward;

  double    nextDist = GetDist(dst,sink);
  int       nextDb   = DistBucket(nextDist);
  WaterCond nextWc   = (did<NUM_NODES)?g_waterCond[did]:CLEAR;
  UpdateQ((int)sid,distB,(int)wLink,action,reward,(int)did,nextDb,(int)nextWc);

  if (g_qcsv.is_open()) {
    auto& qq = GetQ((int)sid,distB,(int)wLink);
    g_qcsv << id << "," << now << "," << sid << "," << did << ","
           << distB << "," << WaterName[wLink] << ","
           << std::setprecision(4)
           << std::get<0>(qq) << "," << std::get<1>(qq) << ","
           << std::get<2>(qq) << ","
           << (action==1?"OPTICAL":action==0?"ACOUSTIC":"DECOY") << ","
           << g_epsilon << "\n";
  }

  Ptr<AquaSimNetDevice> dev = DynamicCast<AquaSimNetDevice>(src->GetDevice(0));
  if (!dev) { g_dropCnt++; return; }

  std::string modeStr;

  if (action==1) {
    dev->GetPhy()->SetTransRange(OPT_RANGE);
    g_optCnt++; g_nstat[sid].optSent++; g_nstat[sid].energy-=OPT_ENERGY;
    g_wcOpt[(int)wLink]++;
    modeStr = "OPTICAL";
    EnemyDetects(sid, OPTICAL_SIGNATURE, "OPTICAL");
    std::cout << "  +------------------------------------------+\n"
              << "  |  >>> OPTICAL (STEALTH) TRANSMISSION <<<  |\n"
              << "  |  dist=" << std::setprecision(1) << dist
              << "m  LQ=" << std::setprecision(3) << optLQ
              << "  eCost=" << OPT_ENERGY << "J         |\n"
              << "  |  Sonar sig=" << OPTICAL_SIGNATURE
              << " freq=500THz — sonar-invisible |\n"
              << "  +------------------------------------------+\n";

  } else if (action==0) {
    dev->GetPhy()->SetTransRange(ACU_RANGE);
    g_acuCnt++; g_nstat[sid].acuSent++; g_nstat[sid].energy-=ACU_ENERGY;
    g_wcAcu[(int)wLink]++;
    modeStr = "ACOUSTIC";
    EnemyDetects(sid, ACOUSTIC_SIGNATURE, "ACOUSTIC");
    std::cout << "  [ ACOUSTIC dist=" << std::setprecision(1) << dist
              << "m  LQ=" << std::setprecision(3) << acuLQ
              << "  freq=" << FREQ_ACOUSTIC << "kHz"
              << "  eCost=" << ACU_ENERGY << "J"
              << (enemyNearby?" *** RISKY — ENEMY NEARBY ***":"") << " ]\n";

  } else {
    dev->GetPhy()->SetTransRange(ACU_RANGE);
    g_decoyCnt++; g_nstat[sid].decoySent++; g_nstat[sid].energy-=DECOY_ENERGY;
    g_wcDecoy[(int)wLink]++;
    modeStr = "DECOY";
    EnemyDetects(sid, DECOY_SIGNATURE, "DECOY");
    std::cout << "  +=============================================+\n"
              << "  |  <<< DECOY BURST — Node" << sid << " >>>           |\n"
              << "  |  Acoustic: fake sonar burst (FOTD style)   |\n"
              << "  |  Optical:  camera flash if enemy <80m      |\n"
              << "  |  freq=" << FREQ_DECOY << "kHz  sig=" << DECOY_SIGNATURE
              << "  eCost=" << DECOY_ENERGY << "J        |\n"
              << "  +=============================================+\n";

    if (g_decoycsv.is_open())
      g_decoycsv << id << "," << now << "," << sid << ","
                 << WaterName[wLink] << "," << std::setprecision(2) << reward << ","
                 << g_enemy[0].isDeceived << "," << g_enemy[1].isDeceived << ","
                 << g_enemy[0].decoySuspicion[sid] << ","
                 << g_enemy[1].decoySuspicion[sid] << "\n";
  }

  std::cout << "  Reward=" << std::setprecision(2) << reward << "\n";

  Ptr<Packet> pkt = Create<Packet>(100);
  bool ok = dev->Send(pkt, AquaSimAddress::GetBroadcast(), 0);
  if (!ok) g_dropCnt++;

  g_step++;
  if (g_step%5==0)
    g_epsilon = std::max(EPS_MIN, g_epsilon*EPS_DECAY);

  if (g_csv.is_open())
    g_csv << id << "," << now << "," << sid << "," << did << ","
          << modeStr << "," << dist << ","
          << std::setprecision(3) << optLQ << "," << acuLQ << ","
          << WaterName[wLink] << "," << reward << ","
          << g_nstat[sid].energy << ","
          << (enemyNearby?"1":"0") << "\n";

  if (id%20==0)
    std::cout << "\n┌─ Snapshot @ pkt " << id << " ─────────────────────────────┐\n"
              << "│  Steps=" << g_step
              << "  Explore=" << g_exploreN
              << "  Exploit=" << g_exploitN
              << "  ε=" << std::setprecision(4) << g_epsilon << "\n"
              << "│  TotalRew=" << std::setprecision(1) << g_totalRew
              << "  States=" << g_qtable.size()
              << "  Opt%=" << std::setprecision(1)
              << (g_optCnt+g_acuCnt+g_decoyCnt>0
                  ?100.0*g_optCnt/(g_optCnt+g_acuCnt+g_decoyCnt):0.0)<<"%\n"
              << "│  Opt=" << g_optCnt << "  Acu=" << g_acuCnt
              << "  Decoy=" << g_decoyCnt << "\n"
              << "│  EnemyDetect=" << (g_enemy[0].realDetections+g_enemy[1].realDetections)
              << "  DecoyHits=" << (g_enemy[0].decoyHits+g_enemy[1].decoyHits)
              << "  OptBlind=" << (g_enemy[0].opticalBlindEvents+g_enemy[1].opticalBlindEvents) << "\n"
              << "└─────────────────────────────────────────────────┘\n";
}

int main(int argc, char* argv[]) {
  double simTime=600.0, interval=1.0;
  CommandLine cmd;
  cmd.AddValue("simTime",  "Simulation duration (s)", simTime);
  cmd.AddValue("interval", "Packet interval (s)",     interval);
  cmd.Parse(argc,argv);

  g_csv.open("hybrid_drl_results.csv");
  g_csv << "PktID,Time,Src,Dst,Mode,Dist,OptLQ,AcuLQ,Water,Reward,Energy,EnemyNearby\n";

  g_qcsv.open("hybrid_qvalues.csv");
  g_qcsv << "PktID,Time,Src,Dst,DistBucket,Water,Qopt,Qacu,Qdecoy,Action,Epsilon\n";

  g_decoycsv.open("hybrid_decoy_log.csv");
  g_decoycsv << "PktID,Time,DecoyNode,Water,Reward,Enemy0Deceived,Enemy1Deceived,"
             << "Suspicion_E0,Suspicion_E1\n";

  InitEnemies();

  std::string d65(65,'=');
  std::cout << "\n" << d65
            << "\n  R-AML-UAN: HYBRID DRL + UDDS + OPTICAL SENSOR [FIXED v3]"
            << "\n" << d65
            << "\n\n  TUNING CHANGES (v3):"
            << "\n  ENEMY_DETECT_THRESH       : 8.0  -> 4.0  (detects sooner)"
            << "\n  Patrol suspicion decay    : 0.4  -> 0.2  (forgets faster)"
            << "\n  Decoy undecided bonus     : 5.0  -> 7.0  (more incentive)"
            << "\n\n  EXPECTED IMPROVEMENTS:"
            << "\n  Decoy Success Rate : 30-50% -> 50-70%"
            << "\n  Stealth Score      : 40-60% -> 60-80%"
            << "\n  Real Detections    : <200   -> <100"
            << "\n\n";

  NodeContainer nodes;
  nodes.Create(NUM_NODES);
  g_nodesPtr = &nodes;

  struct NodePos { double x,y; };
  NodePos layout[NUM_NODES] = {
    { 50.0, 200.0},
    {125.0, 200.0},
    {327.0, 200.0},
    {417.0, 200.0},
    {567.0, 200.0},
    {652.0, 200.0},
  };

  Ptr<ListPositionAllocator> pos=CreateObject<ListPositionAllocator>();
  for (uint32_t i=0;i<NUM_NODES;i++)
    pos->Add(Vector(layout[i].x,layout[i].y,0));

  MobilityHelper mob;
  mob.SetPositionAllocator(pos);
  mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mob.Install(nodes);

  AquaSimChannelHelper channel=AquaSimChannelHelper::Default();
  channel.SetPropagation("ns3::AquaSimRangePropagation");
  AquaSimHelper asHelper=AquaSimHelper::Default();
  asHelper.SetChannel(channel.Create());
  asHelper.SetMac("ns3::AquaSimBroadcastMac");
  asHelper.SetRouting("ns3::AquaSimRoutingDummy");
  for (uint32_t i=0;i<NUM_NODES;i++) {
    Ptr<AquaSimNetDevice> dev=CreateObject<AquaSimNetDevice>();
    asHelper.Create(nodes.Get(i),dev);
    dev->GetPhy()->SetTransRange(ACU_RANGE);
  }

  uint32_t pairs[5][2]={{0,1},{1,2},{2,3},{3,4},{4,5}};
  Ptr<Node> sink=nodes.Get(NUM_NODES-1);
  for (auto& p : pairs) {
    double t=1.0+p[0]*0.4;
    while (t<simTime-1.0) {
      Simulator::Schedule(Seconds(t),&HandlePacket,
        nodes.Get(p[0]),nodes.Get(p[1]),sink);
      t+=interval;
    }
  }

  Simulator::Schedule(Seconds( 50.0),&ChangeWater,1u,MURKY);
  Simulator::Schedule(Seconds( 80.0),&ChangeWater,3u,PRISTINE);
  Simulator::Schedule(Seconds( 80.0),&ChangeWater,4u,PRISTINE);
  Simulator::Schedule(Seconds(100.0),&ChangeWater,1u,PRISTINE);
  Simulator::Schedule(Seconds(100.0),&ChangeWater,2u,PRISTINE);
  Simulator::Schedule(Seconds(180.0),&ChangeWater,1u,CLEAR);
  Simulator::Schedule(Seconds(180.0),&ChangeWater,2u,MURKY);
  Simulator::Schedule(Seconds(200.0),&ChangeWater,3u,COASTAL);
  Simulator::Schedule(Seconds(200.0),&ChangeWater,4u,MURKY);
  Simulator::Schedule(Seconds(300.0),&ChangeWater,2u,PRISTINE);
  Simulator::Schedule(Seconds(300.0),&ChangeWater,3u,PRISTINE);
  Simulator::Schedule(Seconds(370.0),&ChangeWater,4u,CLEAR);
  Simulator::Schedule(Seconds(420.0),&ChangeWater,2u,HAZY);
  Simulator::Schedule(Seconds(450.0),&ChangeWater,0u,MURKY);
  Simulator::Schedule(Seconds(480.0),&ChangeWater,3u,THERMAL);
  Simulator::Schedule(Seconds(480.0),&ChangeWater,4u,THERMAL);
  Simulator::Schedule(Seconds(540.0),&ChangeWater,0u,CLEAR);
  Simulator::Schedule(Seconds(540.0),&ChangeWater,3u,MURKY);
  Simulator::Schedule(Seconds(540.0),&ChangeWater,4u,CLEAR);

  Simulator::Schedule(Seconds(100.0),&PatrolEnemy,0u,380.0,350.0);
  Simulator::Schedule(Seconds(200.0),&PatrolEnemy,0u,280.0,320.0);
  Simulator::Schedule(Seconds(250.0),&PatrolEnemy,0u,130.0,220.0);
  Simulator::Schedule(Seconds(300.0),&PatrolEnemy,0u,180.0,360.0);
  Simulator::Schedule(Seconds(400.0),&PatrolEnemy,0u,480.0,400.0);
  Simulator::Schedule(Seconds(450.0),&PatrolEnemy,0u,390.0,220.0);
  Simulator::Schedule(Seconds(500.0),&PatrolEnemy,0u,340.0,310.0);

  Simulator::Schedule(Seconds( 50.0),&PatrolEnemy,1u,600.0,380.0);
  Simulator::Schedule(Seconds(150.0),&PatrolEnemy,1u,540.0,310.0);
  Simulator::Schedule(Seconds(250.0),&PatrolEnemy,1u,440.0,340.0);
  Simulator::Schedule(Seconds(350.0),&PatrolEnemy,1u,630.0,195.0);
  Simulator::Schedule(Seconds(450.0),&PatrolEnemy,1u,490.0,300.0);
  Simulator::Schedule(Seconds(520.0),&PatrolEnemy,1u,570.0,215.0);

  AnimationInterface anim("hybrid-animation.xml");
  g_anim=&anim;
  anim.EnablePacketMetadata(true);
  for (uint32_t i=0;i<NUM_NODES;i++)
    anim.SetConstantPosition(nodes.Get(i),layout[i].x,layout[i].y);

  anim.UpdateNodeDescription(nodes.Get(0),"N0-SOURCE");
  anim.UpdateNodeDescription(nodes.Get(1),"N1-RELAY");
  anim.UpdateNodeDescription(nodes.Get(2),"N2-RELAY");
  anim.UpdateNodeDescription(nodes.Get(3),"N3-RELAY");
  anim.UpdateNodeDescription(nodes.Get(4),"N4-RELAY");
  anim.UpdateNodeDescription(nodes.Get(5),"N5-SINK");

  for (uint32_t i=0; i<NUM_NODES; i++) {
    SetWaterColor(&anim, nodes.Get(i), g_waterCond[i]);
    double sz=(i==0||i==NUM_NODES-1)?25.0:20.0;
    anim.UpdateNodeSize(nodes.Get(i),sz,sz);
  }

  std::cout << d65 << "\n  SIMULATION RUNNING (" << simTime << "s)\n"
            << d65 << "\n\n";
  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
  Simulator::Destroy();
  g_anim=nullptr; g_nodesPtr=nullptr;

  uint32_t total=g_optCnt+g_acuCnt+g_decoyCnt;
  std::cout << "\n" << d65 << "\n  FINAL REPORT\n" << d65 << "\n\n";

  std::cout << "  Total    : " << total << "\n"
            << "  Optical  : " << g_optCnt << "  ("
            << std::fixed << std::setprecision(1)
            << (total>0?100.0*g_optCnt/total:0.0) << "%)\n"
            << "  Acoustic : " << g_acuCnt << "  ("
            << (total>0?100.0*g_acuCnt/total:0.0) << "%)\n"
            << "  Decoy    : " << g_decoyCnt << "  ("
            << (total>0?100.0*g_decoyCnt/total:0.0) << "%)\n"
            << "  Dropped  : " << g_dropCnt << "\n\n";

  if (total>0) {
    int ob=(int)(40.0*g_optCnt/total);
    int ab=(int)(40.0*g_acuCnt/total);
    int db=(int)(40.0*g_decoyCnt/total);
    std::cout << "  OPTICAL  [" << std::string(ob,'#') << std::string(40-ob,' ')
              << "] " << 100.0*g_optCnt/total << "%\n"
              << "  ACOUSTIC [" << std::string(ab,'#') << std::string(40-ab,' ')
              << "] " << 100.0*g_acuCnt/total << "%\n"
              << "  DECOY    [" << std::string(db,'#') << std::string(40-db,' ')
              << "] " << 100.0*g_decoyCnt/total << "%\n\n";
  }

  std::cout << "  DRL STATS\n" << std::string(40,'-') << "\n"
            << "  Steps    : " << g_step << "\n"
            << "  Explore  : " << g_exploreN << "\n"
            << "  Exploit  : " << g_exploitN << "\n"
            << "  Final e  : " << std::setprecision(4) << g_epsilon << "\n"
            << "  TotalRew : " << std::setprecision(1) << g_totalRew << "\n"
            << "  AvgRew   : " << (g_step>0?g_totalRew/g_step:0.0) << "\n"
            << "  Q-States : " << g_qtable.size() << "\n\n";

  std::cout << "  PER-NODE STATS\n" << std::string(40,'-') << "\n";
  for (uint32_t i=0;i<NUM_NODES;i++)
    std::cout << "  Node" << i
              << " (" << WaterName[g_waterCond[i]] << ")"
              << "  Opt="   << g_nstat[i].optSent
              << "  Acu="   << g_nstat[i].acuSent
              << "  Decoy=" << g_nstat[i].decoySent
              << "  E="     << std::setprecision(1) << g_nstat[i].energy << "J\n";

  std::cout << "\n  WATER CONDITION STATS\n" << std::string(40,'-') << "\n";
  for (int w=0;w<7;w++)
    std::cout << "  " << WaterName[w]
              << "  Opt="   << g_wcOpt[w]
              << "  Acu="   << g_wcAcu[w]
              << "  Decoy=" << g_wcDecoy[w] << "\n";

  uint32_t totDet=0,totDec=0,totMiss=0;
  for (uint32_t e=0;e<NUM_ENEMY;e++){
    totDet +=g_enemy[e].realDetections;
    totDec +=g_enemy[e].decoyHits;
    totMiss+=g_enemy[e].missedReal;
  }

  std::cout << "\n  UDDS RESULTS (Research-Level)\n" << std::string(40,'-') << "\n";
  for (uint32_t e=0;e<NUM_ENEMY;e++) {
    std::cout << "  Enemy-" << e
              << "  RealDetected=" << g_enemy[e].realDetections
              << "  DecoyHits="    << g_enemy[e].decoyHits
              << "  Missed="       << g_enemy[e].missedReal << "\n";
    std::cout << "    Final suspicion per node: [";
    for (int i=0;i<6;i++)
      std::cout << std::setprecision(1) << g_enemy[e].decoySuspicion[i]
                << (i<5?",":"");
    std::cout << "]\n";
  }

  std::cout << "\n  OPTICAL SENSOR RESULTS\n" << std::string(40,'-') << "\n";
  for (uint32_t e=0;e<NUM_ENEMY;e++)
    std::cout << "  Enemy-" << e
              << "  OpticalDetections=" << g_enemy[e].opticalDetections
              << "  OpticalBlindEvents=" << g_enemy[e].opticalBlindEvents
              << "  CurrentlyBlinded="  << (g_enemy[e].isOpticalBlinded?"YES":"NO")
              << "\n";

  double ss=(totDet+totDec>0)?100.0*totDec/(totDet+totDec):0.0;
  std::cout << "\n  TOTAL Real Detections : " << totDet  << "  (lower=better)\n"
            << "  TOTAL Decoy Hits      : " << totDec  << "  (higher=better)\n"
            << "  TOTAL Enemy Missed    : " << totMiss << "\n"
            << "  Decoy Activations     : " << g_decoyCnt << "\n"
            << "  Decoy Success Rate    : "
            << (g_decoyCnt>0?100.0*g_decoySuccess/g_decoyCnt:0.0) << "%\n"
            << "  STEALTH SCORE         : " << std::setprecision(1) << ss << "%\n\n";

  std::cout << "  Q-TABLE (3-action: Qopt/Qacu/Qdecoy)\n"
            << std::string(55,'-') << "\n";
  for (auto& kv : g_qtable) {
    int ni=std::get<0>(kv.first);
    int db=std::get<1>(kv.first);
    int wc=std::get<2>(kv.first);
    double qo=std::get<0>(kv.second);
    double qa=std::get<1>(kv.second);
    double qd=std::get<2>(kv.second);
    std::string best = (qo>=qa&&qo>=qd)?"OPTICAL":(qa>=qo&&qa>=qd)?"acoustic":"DECOY";
    std::cout << "  Node=" << ni << " distB=" << db << " water=" << WaterName[wc]
              << "  Qopt=" << std::setprecision(2) << qo
              << "  Qacu=" << qa << "  Qdec=" << qd
              << "  --> " << best << "\n";
  }

  std::cout << "\n  CHANGES APPLIED (v3)\n" << std::string(55,'-') << "\n"
            << "  ENEMY_DETECT_THRESH  : 8.0 -> 4.0 (enemy detects sooner)\n"
            << "  Suspicion decay rate : 0.4 -> 0.2 (forgets twice as fast)\n"
            << "  Decoy undecided bonus: 5.0 -> 7.0 (stronger decoy incentive)\n\n";

  std::cout << "  SIMULATION COMPLETE\n" << d65 << "\n"
            << "  NetAnim XML  : hybrid-animation.xml\n"
            << "  Main CSV     : hybrid_drl_results.csv\n"
            << "  Q-value CSV  : hybrid_qvalues.csv\n"
            << "  Decoy log    : hybrid_decoy_log.csv\n"
            << d65 << "\n\n";

  g_csv.close(); g_qcsv.close(); g_decoycsv.close();
  return 0;
}
