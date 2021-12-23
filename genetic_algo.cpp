#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>

using std::cin;
using std::cout;
using std::vector;
using std::max;
using std::min;
using std::sort;
using std::string;
using std::thread;

// ================Structures===================

struct Point {
    // x座標
    int x;
    // y座標
    int y;
    Point() {}
    Point(int x, int y) : x(x), y(y) {}
    // pt1及pt2的距離
    static int distOf(const Point&, const Point&);
};

typedef struct RetailStore {
    // 位置座標
    Point pos;
    // 編號i
    int i_number;
    // 需求量D_i
    int D_demand;
    // 每日建設成本f_i
    int f_dailyCnstrctnCst;
    // 單位商品價錢P_i
    int p_unitPrice;
    // 是否營業
    bool isOperating;
    // 是否已被滿足
    bool isSatisfied;
    // 毛收入
    int TR_totalRevenue;
    // 所有物流中心dc_j對自己rs_i的補貨量
    vector<int> restockCnts;

    RetailStore() {}
    RetailStore(Point pt, int i, int dstrbntCntrCnt) : pos(pt), i_number(i), TR_totalRevenue(0), isOperating(true), isSatisfied(false) {
        restockCnts = vector<int>(dstrbntCntrCnt);
        fill(restockCnts.begin(), restockCnts.end(), 0);
    }

    int getTotalStockCnt() const;

} RS;

typedef struct DistributionCenter {
    // 位置座標
    Point pos;
    // 編號j
    int j_number;
    // 每日設置成本h_j
    int h_dailyCnfigCst;
    // 最大庫存量K_j
    int K_maxCapacity;
    // 目前剩餘庫存inv_j
    int inv_currInv;
    // 物流中心透過補貨所賺取的總淨利(pi)
    int TR_totalRevenue;
    // 是否營業
    bool isOperating;

    DistributionCenter() {}
    DistributionCenter(Point pt, int j) : pos(pt), j_number(j), TR_totalRevenue(0), isOperating(true) {}
    // 物流中心j到零售商店i的距離d_ij
    int distTo(const RS&) const;

} DC;

typedef struct Option {

    float H_LAMBDA_S1;
    float H_LAMBDA_S2;
    float F_LAMBDA_S1;
    float F_LAMBDA_S2;

    // 此選項的補貨零售商店編號i
    int i_rsNumber;
    // 此選項的補貨中心編號j
    int j_dcNumber;
    // 此選項的補貨量
    int x_restockCnt;
    // 此選項的單位淨利M
    int M_unitNetProfit;
    // 此選項的總毛利
    int TR_totalRevenue;
    // 此選項的零售商店平均每日建設成本
    int f_dailyCnstrctnCst;
    // 此選項的物流中心平均每日設置成本
    int h_dailyCnfigCst;
    // 此選項的物流中心最大容量
    int K_maxCapacity;

    Option(int i, int j, int M, int x, int f, int h, float hs1, float hs2, float fs1, float fs2)
    : i_rsNumber(i), j_dcNumber(j), x_restockCnt(x), M_unitNetProfit(M), 
    TR_totalRevenue(M*x), f_dailyCnstrctnCst(f), h_dailyCnfigCst(h),
    H_LAMBDA_S1(hs1), H_LAMBDA_S2(hs2), F_LAMBDA_S1(fs1), F_LAMBDA_S2(fs2) {}
    Option() {}
    static Option bestOS1(const Option&, const Option&);
    static Option bestOS2(const Option&, const Option&);
} O;

struct Case {
    int n, m, c, s;
    vector<Point> dcPos;
    vector<Point> rsPos;
    vector<int> ds;
    vector<int> fs;
    vector<int> ps;
    vector<int> hs;
    vector<int> Ks;
    Case(int n, int m, int c, int s,
         vector<Point> dcPos,
         vector<Point> rsPos,
         vector<int> ds,
         vector<int> fs,
         vector<int> ps,
         vector<int> hs,
         vector<int> Ks) :
    n(n), m(m), c(c), s(s), 
    dcPos(dcPos),
    rsPos(rsPos),
    ds(ds),
    fs(fs),
    ps(ps),
    hs(hs),
    Ks(Ks) {}

    Case() {

    }
};

const int MINn = 1;
const int MAXn = 50;
const int MINm = 1;
const int MAXm = 1000;
const int MINc = 1;
const int MAXc = 10;
const int MINs = 1;
const int MAXs = 2;
const int MINx = 0;
const int MAXx = 200;
const int MINy = 0;
const int MAXy = 200;
const int MINu = 0;
const int MAXu = 200;
const int MINv = 0;
const int MAXv = 200;
const int MIND = 1;
const int MAXD = 1000;
const int MINf = 1;
const int MAXf = 1e6;
const int MINp = 1;
const int MAXp = 500;
const int MINh = 1;
const int MAXh = 1e7;
const int MINK = 1;
const int MAXK = 1e5;

const int MINtest = 1;
const int MAXtest = 25;

const int S[MAXtest] = {
        1,   1,   1,   1,   1,
        1,   1,   1,   1,   1,
        2,   2,   2,   2,   2,
        2,   2,   2,   2,   2,
        1,   1,   2,   2,   2
};
const int N[MAXtest] = {
        1,   2,   3,   4,   5,
       10,  15,  20,  30,  40,
        2,   3,   5,   7,   9,
       10,  15,  20,  30,  40,
       50,  50,  50,  50,  50
};
const int M[MAXtest] = {
       5,  10,  30,  60, 100,
     200, 400, 600, 800,1000,
       5,  10,  30,  60, 100,
     200, 400, 600, 800, 987,
    1000,1000,1000,1000,1000
};
const int C[MAXtest] = {
        2, 1, 1, 1, 1,
        1, 2, 1, 1, 1,
        1, 1, 2, 1, 1,
        1, 1, 1, 2, 1,
        1, 1, 1, 1, 2
};
long long int magicNumber = 0;
const long long int BASE = 8705029; // Randomly pick a prime number
const long long int MOD = 1e11 + 3; // Yet another prime number
const long long int OFFSET =
( static_cast<long long int>('I')
+ static_cast<long long int>(' ')
+ static_cast<long long int>('<')
+ static_cast<long long int>('3')
+ static_cast<long long int>(' ')
+ static_cast<long long int>('P')
+ static_cast<long long int>('r')
+ static_cast<long long int>('o')
+ static_cast<long long int>('g')
+ static_cast<long long int>('r')
+ static_cast<long long int>('a')
+ static_cast<long long int>('m')
+ static_cast<long long int>('m')
+ static_cast<long long int>('i')
+ static_cast<long long int>('n')
+ static_cast<long long int>('g')
+ static_cast<long long int>(' ')
+ static_cast<long long int>('D')
+ static_cast<long long int>('e')
+ static_cast<long long int>('s')
+ static_cast<long long int>('i')
+ static_cast<long long int>('g')
+ static_cast<long long int>('n')
+ static_cast<long long int>('!')
)
*
( static_cast<long long int>('R')
+ static_cast<long long int>('o')
+ static_cast<long long int>('c')
+ static_cast<long long int>('k')
+ static_cast<long long int>(' ')
+ static_cast<long long int>('&')
+ static_cast<long long int>(' ')
+ static_cast<long long int>('r')
+ static_cast<long long int>('r')
+ static_cast<long long int>('r')
+ static_cast<long long int>('o')
);

// RNG is the abbreviation of "Random Number Generator"
long long int naiveRNG(long long int l, long long int r) {
    magicNumber %= MOD;
    magicNumber = (BASE * magicNumber + OFFSET) % MOD;
    return magicNumber % (r - l + 1) + l;
}

Case generateCase(int n, int m, int c, int s){

    vector<Point> dcPos(n);
    for (int i = 0; i < n; i++)
        dcPos[i] = Point((int) naiveRNG(MINu, MAXu), (int) naiveRNG(MINv, MAXv));
    
    vector<Point> rsPos(m);
    for (int i = 0; i < m; i++)
        rsPos[i] = Point((int) naiveRNG(MINx, MAXx), (int) naiveRNG(MINx, MAXx));

    vector<int> ds(m);
    for (int i = 0; i < m; i++)
        ds[i] = (int) naiveRNG(MIND, MAXD);
    
    vector<int> fs(m);
    for (int i = 0; i < m; i++)
        fs[i] = (int) naiveRNG(MINf, MAXf);

    vector<int> ps(m);
    for (int i = 0; i < m; i++)
        ps[i] = (int) naiveRNG(MINp, MAXp);

    vector<int> hs(n);
    for (int i = 0; i < n; i++)
        hs[i] = (int) naiveRNG(MINh, MAXh);

    vector<int> Ks(n);
    for (int i = 0; i < n; i++)
        Ks[i] = (int) naiveRNG(MINK, MAXK);

    return Case(n, m, c, s,
                dcPos,
                rsPos,
                ds,
                fs,
                ps,
                hs,
                Ks);
}

// ================Classes===================

class Retailer {
public:

    float H_LAMBDA_S1;
    float H_LAMBDA_S2;
    float F_LAMBDA_S1;
    float F_LAMBDA_S2;
    // 接收測資
    void getInput(Case);
    // 印出物件內部狀態
    void printState() const;
    // 外部介面：一鍵解題
    void solve();
    
    int getFinalNetProfit() const;
    // debug專用，顯示所有零售商店資訊
    void __debug_displayRss() const;
    // debug專用，顯示所有物流中心資訊
    void __debug_displayDcs() const;
    // debug專用，顯示補貨情況是否符合題目限制。
    void __debug_displayValidity() const;
    // debug專用，顯示總體淨利潤
    void __debug_displayFinalNetProfit() const;
    // debug專用，顯示所有資訊
    void __debug_displayAll() const;

    Retailer(float hs1, float hs2,
             float fs1, float fs2)
    : H_LAMBDA_S1(hs1), H_LAMBDA_S2(hs2),
    F_LAMBDA_S1(fs1), F_LAMBDA_S2(fs2) {} 

private:

    // 每單位補貨的每公里成本c
    int c_restockCostPerKm;
    // 單或多中心原則s(false/true)
    bool s_isMultipleCenter;
    // 零售商店個數m
    int m_rsCnt;
    // 物流中心個數n
    int n_dcCnt;

    // 零售商店列表
    vector<RS> rss_retailStores;
    // 物流中心列表
    vector<DC> dcs_dstrbtnCntrs;
    // 關閉給並編號之零售商店
    void __internal_shutDownRs(int);
    // 關閉給並編號之物流中心
    void __internal_shutDownDc(int);
    // 關閉賠錢的物流中心
    void __internal_shutDownDeficitDcs();
    // 關閉賠錢的零售商店
    void __internal_shutDownDeficitRss();
    // 透過O物件補貨
    void __internal_restock(const O&);

    void __internal_reset();
    // 取得根據給定rs及dc製作的Option物件
    O getOption(const RS&, const DC&) const;
    // 取得目前最佳O
    O getBestO() const;
    // 最佳化
    void __internal_optimize();
};

// 加總傳入的整數vector
int sum(vector<int>);

// ===============Main Function=================

#include <ctime>
#include <random>

// Number of individuals in each generation
#define POPULATION_SIZE 10

// Valid Genes
vector<float> GENES(20);


// Function to generate random numbers in given range
int random_num(int start, int end) {
	int range = end - start + 1;
	int random_int = start + (rand() % range);
	return random_int;
}

// Create random genes for mutation
float mutated_genes() {
	int len = GENES.size();
	int r = random_num(0, len-1);
	return GENES[r];
}

// create chromosome or string of genes
vector<float> create_gnome() {
	vector<float> gnome(4);
	for(int i = 0; i < 4; ++i)
		gnome[i] = mutated_genes();
	return gnome;
}

// Class representing individual in population
class Individual {
public:
	vector<float> chromosome;
	long long fitness;
    Individual() {}
	Individual(vector<float> chromosome);
	Individual mate(Individual parent2);
	long long cal_fitness();
};

Individual::Individual(vector<float> chromosome) {
	this->chromosome = chromosome;
	fitness = cal_fitness();
};

// Perform mating and produce new offspring
Individual Individual::mate(Individual par2) {
	// chromosome for offspring
	vector<float> child_chromosome;

	int len = chromosome.size();
	for(int i = 0; i < len; ++i) {
		// random probability
		float p = random_num(0, 100) / 100;

		// if prob is less than 0.45, insert gene
		// from parent 1
		if(p < 0.45f)
			child_chromosome.push_back(chromosome[i]);

		// if prob is between 0.45 and 0.90, insert
		// gene from parent 2
		else if(p < 0.9f)
			child_chromosome.push_back(par2.chromosome[i]);

		// otherwise insert random gene(mutate),
		// for maintaining diversity
		else
			child_chromosome.push_back(mutated_genes());
	}

	// create new Individual(offspring) using
	// generated chromosome for offspring
	return Individual(child_chromosome);
};

Case testCases[25];
long long Individual::cal_fitness() {
    long long fitness = 0;
    for (int level = 0; level < 25; ++level) {
        Retailer retailer(chromosome[0], chromosome[1], chromosome[2], chromosome[3]);
        retailer.getInput(testCases[level]);
        retailer.solve();
        fitness += retailer.getFinalNetProfit();
    }
	return fitness;
};


// Driver code
int main() {

	srand( time(nullptr) );
    magicNumber = 322;
    cout << "[System] Building test cases...\n";
    for (int level = 0; level < 25; ++level) {
        testCases[level] = generateCase(
                        N[level],
                        M[level],
                        C[level],
                        S[level]
        );
    }
    cout << "[System] Test cases built successfully.\n";

    // build gene pool
    for (int i = 1; i <= 20; ++i)
        GENES[i-1] = .05f * i;

	// current generation
	int generation = 0;

	bool breakpoint = false;
	// create initial population

	vector<Individual> population(POPULATION_SIZE);
    vector<thread> threads;
    cout << "[System] Forming individual of the initial population...\n";
	for (int i = 0; i < POPULATION_SIZE; ++i)
		threads.push_back(
            thread(
                [](vector<Individual>& population, int i) {
                    population[i] = Individual(create_gnome());
                },
                std::ref(population),
                i
            )
        );
    for (int i = 0; i < POPULATION_SIZE; ++i)
        threads[i].join();

    cout << "[System] Initial population formed successfully.\n";

	while (!breakpoint) {

		// sort the population in increasing order of fitness score
		sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });

		if (generation >= 2000) {
			breakpoint = true;
			break;
		}

		// Perform Elitism, that mean 10% of fittest population
		// goes to the next generation
		int s = (10 * POPULATION_SIZE) / 100;
		// Otherwise generate new offsprings for new generation
		vector<Individual> new_generation(s);
		for(int i = 0; i < s; ++i)
			new_generation[i] = population[i];

		// From 40% of fittest population, Individuals
		// will mate to produce offspring
		s = (90 * POPULATION_SIZE) / 100;
		for(int i = 0; i < s; i++) {
            cout << "[System] Reproducing individual " << i << " of the new generation...\n";
			int len = population.size();
			int r = random_num(0, (int) (.4f * len));
			Individual parent1 = population[r];
			r = random_num(0, (int) (.4f * len));
			Individual parent2 = population[r];
			Individual offspring = parent1.mate(parent2);
			new_generation.push_back(offspring);
            cout << "[System] Individual " << i << " of the new generation reproduced successfully. (" << i+1 << "/" << s << ")\n";
		}
		population = new_generation;
		cout<< "Generation: " << generation << "\t";
        for (auto param : population[0].chromosome) 
            cout << param << " ";
		cout<< "Fitness: "<< population[0].fitness << "\n";

		++generation;
	}
	cout<< "Generation: " << generation << "\t";
    for (auto param : population[0].chromosome) 
        cout << param << " ";
	cout<< "Fitness: "<< population[0].fitness << "\n";
}


// ===========Def of Functions==============

int sum(vector<int> vec) {
    int _sum = 0, len = vec.size();
    for (int i = 0; i < len; ++i) _sum += vec[i];
    return _sum;
}   

// ===========Def of Compare Functions==============

O O::bestOS1(const O& o1, const O& o2) {
    return (o1.TR_totalRevenue - o1.f_dailyCnstrctnCst * o1.F_LAMBDA_S1 - o1.h_dailyCnfigCst* o1.H_LAMBDA_S1 > o2.TR_totalRevenue - o2.f_dailyCnstrctnCst* o2.F_LAMBDA_S1 - o2.h_dailyCnfigCst* o2.H_LAMBDA_S1) ? o1 : o2;
}

O O::bestOS2(const O& o1, const O& o2) {
    return (o1.M_unitNetProfit - (o1.f_dailyCnstrctnCst * o1.F_LAMBDA_S2 + o1.h_dailyCnfigCst * o1.H_LAMBDA_S2) / o1.x_restockCnt > o2.M_unitNetProfit - (o2.f_dailyCnstrctnCst * o2.F_LAMBDA_S2 + o2.h_dailyCnfigCst * o2.H_LAMBDA_S2) / o2.x_restockCnt) ? o1 : o2;
}

// ===========Def of Methods==============

int Point::distOf(const Point& pt1, const Point& pt2) {
    return abs(pt1.x - pt2.x) + abs(pt1.y - pt2.y); 
}

int RS::getTotalStockCnt() const {
    return sum(restockCnts);
}

int DC::distTo(const RS& rs) const {
    return Point::distOf(this->pos, rs.pos); 
}

void Retailer::getInput(Case testCase) {
   m_rsCnt = testCase.m;
   n_dcCnt = testCase.n;
   c_restockCostPerKm = testCase.c;
   s_isMultipleCenter = testCase.s - 1;

    dcs_dstrbtnCntrs = vector<DC>(n_dcCnt);
    rss_retailStores = vector<RS>(m_rsCnt);

    for (int j = 0; j < n_dcCnt; ++j)
        dcs_dstrbtnCntrs[j] = DC(testCase.dcPos[j], j+1);

    for (int i = 0; i < m_rsCnt; ++i)
        rss_retailStores[i] = RS(testCase.rsPos[i], i+1, n_dcCnt);

    for (int i = 0; i < m_rsCnt; ++i)
        rss_retailStores[i].D_demand = testCase.ds[i];

    for (int i = 0; i < m_rsCnt; ++i)
        rss_retailStores[i].f_dailyCnstrctnCst = testCase.fs[i];

    for (int i = 0; i < m_rsCnt; ++i)
        rss_retailStores[i].p_unitPrice = testCase.ps[i];

    for (int j = 0; j < n_dcCnt; ++j)
        dcs_dstrbtnCntrs[j].h_dailyCnfigCst = testCase.hs[j];

    for (int j = 0; j < n_dcCnt; ++j) {
        dcs_dstrbtnCntrs[j].K_maxCapacity = testCase.Ks[j];
        dcs_dstrbtnCntrs[j].inv_currInv = testCase.Ks[j];
    }
}

void Retailer::__internal_shutDownRs(int targetI) {
    RS& targetRs = rss_retailStores[targetI - 1];
    targetRs.isOperating = false;
    targetRs.isSatisfied = false;
    targetRs.TR_totalRevenue = 0;
    fill(targetRs.restockCnts.begin(), targetRs.restockCnts.end(), 0);
}

void Retailer::__internal_shutDownDc(int targetJ) {
    DC& targetDc = dcs_dstrbtnCntrs[targetJ - 1];
    targetDc.isOperating = false;
    targetDc.inv_currInv = targetDc.K_maxCapacity;
    targetDc.TR_totalRevenue = 0;
    for (auto& rs : rss_retailStores)
        rs.restockCnts[targetJ - 1] = 0;
}

void Retailer::__internal_shutDownDeficitDcs() {
    for (const auto& dc : dcs_dstrbtnCntrs)
        if (dc.TR_totalRevenue < dc.h_dailyCnfigCst)
            __internal_shutDownDc(dc.j_number);
}

void Retailer::__internal_shutDownDeficitRss() {
    for (const auto& rs : rss_retailStores)
        if (rs.TR_totalRevenue < rs.f_dailyCnstrctnCst)
            __internal_shutDownRs(rs.i_number);
}

O Retailer::getOption(const RS& rs, const DC& dc) const {
    if (rs.isSatisfied || !rs.isOperating)
        return O(rs.i_number, dc.j_number, 0, 0, 0, 0, 0, 0, 0, 0);
    int p = rs.p_unitPrice;
    int d = dc.distTo(rs);
    int x = min(rs.D_demand - rs.getTotalStockCnt(), dc.inv_currInv);
    int c = c_restockCostPerKm;
    int f = rs.getTotalStockCnt() == 0 ? rs.f_dailyCnstrctnCst : 0;
    int h = dc.inv_currInv == dc.K_maxCapacity ? dc.h_dailyCnfigCst : 0;

    return O(rs.i_number, dc.j_number, p - c*d, x, f, h, H_LAMBDA_S1, H_LAMBDA_S2, F_LAMBDA_S1, F_LAMBDA_S2);
}

void Retailer::__internal_restock(const O& option) {
    RS& targetRs = rss_retailStores[option.i_rsNumber-1];
    DC& targetDc = dcs_dstrbtnCntrs[option.j_dcNumber-1];
    targetRs.restockCnts[option.j_dcNumber-1] += option.x_restockCnt;
    targetDc.inv_currInv -= option.x_restockCnt;
    targetDc.TR_totalRevenue += option.TR_totalRevenue;
    targetRs.TR_totalRevenue += option.TR_totalRevenue;
    if (s_isMultipleCenter)
        // s=2
        targetRs.isSatisfied = targetRs.getTotalStockCnt() == targetRs.D_demand;
    else
        // s=1
        targetRs.isSatisfied = true;
}

void Retailer::__internal_reset() {
    for (int i = 0; i < m_rsCnt; ++i)
        __internal_shutDownRs(i+1);
    for (int j = 0; j < n_dcCnt; ++j)
        __internal_shutDownDc(j+1);
}

O Retailer::getBestO() const {
    O bestO = O(0, 0, 0, -1, 0, 0, 0, 0, 0, 0);
    O currO;
    for (const auto& dc : dcs_dstrbtnCntrs)
        for (const auto& rs : rss_retailStores)
            if ((currO = getOption(rs, dc)).x_restockCnt > 0)
                if (!s_isMultipleCenter) 
                    bestO = O::bestOS1(bestO, currO);
                else 
                    bestO = O::bestOS2(bestO, currO);
    return bestO;
}

void Retailer::__internal_optimize() {
    O bestO;
    while ( (bestO = getBestO()).x_restockCnt > 0 )
        __internal_restock(bestO);

    __internal_shutDownDeficitDcs();
    __internal_shutDownDeficitRss();
}

void Retailer::printState() const {
        if (getFinalNetProfit() < 0) {
        cout << "0\n0\n";
        for (int i = 0; i < m_rsCnt; ++i) {
            for (int j = 0; j < n_dcCnt; ++j)
                cout << "0 ";
            cout << "\n";
        }
        return;
    }

    // 列印第一列__internal_輸出：
    // =============================
    // 一個非負整數n bar屬於[0, n]
    // 代表「要設置的物流中心」的個數

    int operatingDcCnt = 0;
    for (int j = 0; j < n_dcCnt; ++j)
        operatingDcCnt += dcs_dstrbtnCntrs[j].isOperating;

    cout << operatingDcCnt;

    // 至多n個屬於[1, n]的不重複整數
    // 代表「要設置的物流中心」編號

    for (int j = 0; j < n_dcCnt; ++j)
        if (dcs_dstrbtnCntrs[j].isOperating)
            cout << " " << dcs_dstrbtnCntrs[j].j_number;

    cout << "\n";

    // 列印第二列輸出：
    // =============================
    // 一個非負整數m bar屬於[0, m]
    // 代表「要設置的零售商店」的個數

    // 至多m個屬於[1, m]的不重複整數
    // 代表「要設置的零售商店」編號

    int operatingRsCnt = 0;
    for (int i = 0; i < m_rsCnt; ++i)
        operatingRsCnt += rss_retailStores[i].isOperating;
    
    cout << operatingRsCnt;

    for (int i = 0; i < m_rsCnt; ++i) 
        if (rss_retailStores[i].isOperating)
            cout << " " << rss_retailStores[i].i_number;
    
    cout << "\n";

    // 列印第3 ~ m+2列輸出：
    // =============================
    for (int i = 0; i < m_rsCnt; ++i) {
        // 對每一個零售商店rs_i印出其自所有物流中心dc_j接受的補貨量x_ij
        for (int j = 0; j < n_dcCnt; ++j) {
            if (rss_retailStores[i].isOperating)
                cout << rss_retailStores[i].restockCnts[j] << " ";
            else
                cout << "0 ";
        }
        cout << "\n";
    }
}

void Retailer::solve() {
    __internal_optimize();

    if (getFinalNetProfit() < 0) __internal_reset();
}

int Retailer::getFinalNetProfit() const {
    int finalNetProfit = 0;

    for (const auto& dc : dcs_dstrbtnCntrs)
        if (dc.isOperating) {
            finalNetProfit += dc.TR_totalRevenue;
            finalNetProfit -= dc.h_dailyCnfigCst;
        }

    for (const auto& rs : rss_retailStores)
        if (rs.isOperating) finalNetProfit -= rs.f_dailyCnstrctnCst;

    return finalNetProfit;
}

// ===========Def of Debug Methods==============

void Retailer::__debug_displayDcs() const {
    for (int j = 0; j < n_dcCnt; ++j) {
        DC currDc = dcs_dstrbtnCntrs[j];
        cout << "第" << currDc.j_number << "號物流中心的資訊\n";
        cout << "==============================================\n";
        cout << "位置座標(x, y)：" << "(" << currDc.pos.x << ", " << currDc.pos.y << ")\n";
        cout << "每日平均設置成本(h)：" << currDc.h_dailyCnfigCst << "\n";
        cout << "庫存量(剩餘inv/最大K)：" << currDc.inv_currInv << " / " << currDc.K_maxCapacity << "\n";
        cout << "營運預期淨利(Tpi, 不含h)：" << currDc.TR_totalRevenue << "\n";
        cout << "是否營運：" << currDc.isOperating << "\n";
        cout << "==============================================\n";
    }
}   

void Retailer::__debug_displayRss() const {
    for (int i = 0; i < m_rsCnt; ++i) {
        RS currRs = rss_retailStores[i];
        cout << "第" << currRs.i_number << "號零售商店的資訊\n";
        cout << "==============================================\n";
        cout << "位置座標(x, y)：" << "(" << currRs.pos.x << ", " << currRs.pos.y << ")\n";
        cout << "當日需求：" << currRs.D_demand << "\n";
        cout << "商品售價：" << currRs.p_unitPrice << "\n";
        cout << "每日平均建設成本：" << currRs.f_dailyCnstrctnCst << "\n";
        cout << "是否營運：" << currRs.isOperating << "\n";
        cout << "來自各物流中心的補貨量(依序為1~n-1)：[";
        for (int j = 0; j < n_dcCnt; ++j) {
            cout << currRs.restockCnts[j];
            if (j < n_dcCnt-1) cout << ", ";
        }
        cout << "]\n";
        cout << "(若營運)是否被滿足：" << currRs.isSatisfied << "\n";
        cout << "==============================================\n";
    }
}

void Retailer::__debug_displayFinalNetProfit() const {
    cout << "預期單日總銷貨淨利：" << getFinalNetProfit() << "\n";
}

void Retailer::__debug_displayValidity() const {
    bool isValid = true;
    long long totalRestockCnt = 0, distributedCnt = 0;
    for (int i = 0; i < m_rsCnt; ++i)
        totalRestockCnt += rss_retailStores[i].getTotalStockCnt() * rss_retailStores[i].isOperating;

    for (int j = 0; j < n_dcCnt; ++j)
        distributedCnt += (dcs_dstrbtnCntrs[j].K_maxCapacity - dcs_dstrbtnCntrs[j].inv_currInv) * dcs_dstrbtnCntrs[j].isOperating;

    if (totalRestockCnt != distributedCnt) {
        cout << "[debug] 總補貨量 != 總發出補貨量\n";
        isValid = false;
    }

    for (int i = 0; i < m_rsCnt; ++i) {
        int restockCnt = sum(rss_retailStores[i].restockCnts);
        if (restockCnt > rss_retailStores[i].D_demand) {
            cout << "[debug] 零售商店" << rss_retailStores[i].i_number << "：補貨量 > 需求量\n";
            isValid = false;
        }
    }

    for (int j = 0; j < n_dcCnt; ++j)
        if (dcs_dstrbtnCntrs[j].inv_currInv < 0) {
            cout << "[debug] 物流中心" << dcs_dstrbtnCntrs[j].j_number << "：出貨量 > 最大容量\n";
            isValid = false;
        } else if (dcs_dstrbtnCntrs[j].inv_currInv > dcs_dstrbtnCntrs[j].K_maxCapacity) {
            cout << "[debug] 物流中心" << dcs_dstrbtnCntrs[j].j_number << "：庫存量 > 最大容量\n";
            isValid = false;
        }

    for (int i = 0; i < m_rsCnt; ++i)
        if (rss_retailStores[i].getTotalStockCnt() == 0 && rss_retailStores[i].isOperating) {
            cout << "[debug] 零售商店" << rss_retailStores[i].i_number << "：赤字營運\n";
            isValid = false;
        }

    for (int j = 0; j < n_dcCnt; ++j)
        if (dcs_dstrbtnCntrs[j].TR_totalRevenue - dcs_dstrbtnCntrs[j].h_dailyCnfigCst < 0 &&
            dcs_dstrbtnCntrs[j].isOperating) {
            cout << "[debug] 物流中心" << dcs_dstrbtnCntrs[j].j_number << "：赤字營運\n";
            isValid = false;
        }


    cout << "[debug] 最終除錯結果：" << (isValid ? "" : "不") << "符合題目要求\n";

}

void Retailer::__debug_displayAll() const {
    // __debug_displayDcs();
    // __debug_displayRss();
    __debug_displayFinalNetProfit();
    // __debug_displayValidity();
}   