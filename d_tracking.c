#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//smapling ruleが実行時間のほとんどを占める。その中でも最適重み(optimal weights)の計算が実行時間のほとんどを占めている。
//最適重みの計算の高速化、もしくは最適重みの計算回数を減らすことができればいい？
//毎回重みを計算せずに、複数回のtrialごとに計算しても精度が保証されたら早いか？

//最適重みの計算方法



//最適重みが時間かかりすぎる

// 変数の手動設定
static bool ratio = true; //sampling ruleの純粋探索を比でおこなうならtrue、従来ならfalse
int iter = 3000; //繰り返し回数
float delta = 0.1; //fixed confidence
float model[4] = {0.5, 0.4, 0.43, 0.4};
//float model[5] = {0.3, 0.21, 0.54, 0.19, 0.5}; //model 2
enum {K = 4}; //アームの数


// 自動決定される必要な変数
int N[K] = {0};//seleted time for each arm

int true_arm = 0;

//報酬
double reward_sum[K];
double reward_mean[K];




int  sampling_rule(int trial, bool ratio, int *num_weight_cal); //所与の条件から次に引くべきアームを選択する。
bool stopping_rule(double threshold); //true:停止、false:続行
int  decision_rule(); //探索結果から最適腕を決定(推定)する。
double bernouii(double p); //選択されたアームの報酬を決める(二値報酬)
double threshold(int trial){
    return log( ( log(trial) + 1) / delta);
}




int main(){
    printf("%15s %15s %15s %15s\n", "sampling(s)", "per exe(s)", "trial(times)", "weight_cal");
double correct_rate = 0;
double ave_spent_trial = 0;

//乱数シードの初期化
srand((unsigned int)time(NULL));

//真の最適腕を調べておく
for(int i=1;i<K;i++){
    if(model[true_arm] < model[i]){
        true_arm = i;
    }
}
//printf("%d\n", true_arm );
clock_t start, end;
clock_t start_sampling, end_sampling;
int clocks = CLOCKS_PER_SEC;
int num_weight_cal_sum=0;

for(int j=0;j<iter;j++){
    start = clock();
    printf("%3d ", j+1);
    fflush(stdout);
    
    //報酬の初期化
    for(int i=0 ; i<K ; i++){
    reward_sum[i] = bernouii(model[i]);
    reward_mean[i] = reward_sum[i];
    N[i] = 1;
    //printf("%3lf %3lf %3lf\n", model1[i], reward_sum[i], reward_mean[i]);
    }
    

    int selected_arm;
    int reward;
    int trial = K-1; //初期化段階ですでにすべてのアームを一回ずつ引いている。

    //printf("%3d", stopping_rule(threshold(trial)));
    start_sampling = clock();
    int num_weight_cal=0;
    while(stopping_rule(threshold(trial)) == false){
        
        trial++;
        
        selected_arm = sampling_rule(trial, ratio, &num_weight_cal);
        
        N[selected_arm] += 1;
        reward = bernouii(model[selected_arm]);
        reward_sum[selected_arm] += reward;
        reward_mean[selected_arm] = reward_sum[selected_arm] / N[selected_arm];

    }
    end_sampling = clock();
    num_weight_cal_sum += num_weight_cal;
    if(decision_rule() == true_arm){
        correct_rate ++;
    }

    ave_spent_trial += trial;

    //printf("Best arm : %d", decision_rule());
    //printf("trial time : %d", trial);
    end = clock();
    //sampling time, execution time, trial, weight calculation time
    printf("%15.3lf(s), %15.3lf(s), %15d(times), %15d\n", (double)(end_sampling - start_sampling) / clocks, (double)(end - start) / clocks, trial, num_weight_cal);

    fflush(stdout);
}
    printf("Ave_weight_cal : %d times\n", num_weight_cal_sum / iter);
    printf("Correct_rate : %3lf\n", correct_rate / iter);
    printf("Average_spent_trial : %3lf\n", ave_spent_trial / iter);
    return 0;
}

double kldiv(double a, double b){
    if(a < 0.0 || 1.0 < a ||
       b < 0.0 || 1.0 < b ||
       isnan(a) || isnan(b)){
        return NAN;
    }else if( a == b ){
        return 0.0;
    }else if( b == 0.0 || b == 1.0){
        return INFINITY;
    }else if( a == 0.0){
        return -log(1-b);
    }else if (a == 1.0){
        return -log(b);
    }else{
        return a * log(a / b) + (1.0-a) * log( (1.0-a) / (1-b) );
    }
}

double weighted_mean(int arm_a, int arm_b){
    return (double)N[arm_a] / (N[arm_a] + N[arm_b]) * reward_mean[arm_a] + (double)N[arm_b] / ( N[arm_a] + N[arm_b]) * reward_mean[arm_b];
}

/* 二分法で f(x) = 0 を解く */
static double bisect(
    double (*f)(double, void *),
    void *ctx,
    double lo,
    double hi,
    double tolsto
) {
    double flo = f(lo, ctx);

    while (hi - lo > tolsto) {
        double mid = (lo + hi) / 2.0;
        double fm = f(mid, ctx);

        if (fm * flo > 0.0) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
    }

    return (lo + hi) / 2.0;
}

/* I(alpha, mu1, mu2) */
static double info_I(double alpha, double a, double b)
{
    if (alpha == 0.0 || alpha == 1.0)
        return 0.0;

    double mid = alpha * a + (1.0 - alpha) * b;

    return alpha * kldiv(a, mid)
         + (1.0 - alpha) * kldiv(b, mid);
}

/* cost(mu1, mu2, nu1, nu2) */
static double cost(
    double a, double b, double nu1, double nu2
) {
    if (nu1 == 0.0 && nu2 == 0.0)
        return 0.0;

    double alpha = nu1 / (nu1 + nu2);

    return (nu1 + nu2) * info_I(alpha, a, b);
}

/* xkofy の二分法用 */
typedef struct {
    double a, b, y;
} XContext;

static double x_equation(double x, void *ptr)
{
    XContext *c = ptr;

    return (1.0 + x) *
        cost(c->a, c->b,
             1.0 / (1.0 + x),
             x / (1.0 + x))
        - c->y;
}

/* x_k(y) */
static double xkofy(double y, double a, double b)
{
    XContext ctx = {a, b, y};

    double hi = 1.0;

    while (x_equation(hi, &ctx) < 0.0) {
        hi *= 2.0;
    }

    return bisect(x_equation, &ctx, 0.0, hi, 1e-11);
}

/* aux(y, mu) */
typedef struct {
    const double *mu;
    int K;
} AuxContext;

static double aux_equation(double y, void *ptr)
{
    AuxContext *c = ptr;

    double a = c->mu[0];
    double sum = 0.0;

    for (int k = 1; k < c->K; k++) {
        double x = xkofy(y, a, c->mu[k]);
        double m = (a + x * c->mu[k]) / (1.0 + x);

        sum += kldiv(a, m) / kldiv(c->mu[k], m);
    }

    return sum - 1.0;
}

/*
 * OptimalWeights の C 版
 *
 * mu      : 各アームの平均報酬（長さ K）
 * K       : アーム数
 * weights : 最適重みの出力先（長さ K）
 *
 * 戻り値  : v = 1 / T*(mu)
 */
double optimal_weights_cal(
    const double *mu,
    double *weights
) {
    const double tol = 1e-11;

    if (mu == NULL || weights == NULL || K <= 0)
        return NAN;

    /* 平均報酬の範囲を検証 */
    for (int i = 0; i < K; i++) {
        if (!isfinite(mu[i]) || mu[i] < 0.0 || mu[i] > 1.0)
            return NAN;
    }

    /* 最大平均報酬を持つアームを数える */
    double max_mu = mu[0];
    int best_count = 0;

    for (int i = 1; i < K; i++) {
        if (mu[i] > max_mu)
            max_mu = mu[i];
    }

    for (int i = 0; i < K; i++) {
        if (mu[i] == max_mu)
            best_count++;
    }

    /* 最大平均報酬が同率の場合 */
    if (best_count > 1) {
        for (int i = 0; i < K; i++)
            weights[i] = (mu[i] == max_mu)
                       ? 1.0 / best_count : 0.0;
        return 0.0;
    }

    if (K == 1) {
        weights[0] = 1.0;
        return 0.0;
    }

    /*
     * 平均報酬を降順に並べる。
     * index は元の配列におけるアーム番号。
     */
    double *sorted = malloc((size_t)K * sizeof(double));
    double *nu = malloc((size_t)K * sizeof(double));
    int *index = malloc((size_t)K * sizeof(int));

    if (sorted == NULL || nu == NULL || index == NULL) {
        free(sorted);
        free(nu);
        free(index);
        return NAN;
    }

    for (int i = 0; i < K; i++) {
        sorted[i] = mu[i];
        index[i] = i;
    }

    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            if (sorted[j] > sorted[i]) {
                double tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;

                int itmp = index[i];
                index[i] = index[j];
                index[j] = itmp;
            }
        }
    }

    /* oneStepOpt: y の上限 */
    AuxContext ctx = {sorted, K};
    double yhi = 0.5;

    if (isinf(kldiv(sorted[0], sorted[1]))) {
        while (aux_equation(yhi, &ctx) < 0.0) {
            yhi *= 2.0;
        }
    } else {
        yhi = kldiv(sorted[0], sorted[1]);
    }

    double y = bisect(
        aux_equation, &ctx, 0.0, yhi, tol
    );

    /* x = [1, x_2, ..., x_K] */
    nu[0] = 1.0;

    for (int i = 1; i < K; i++)
        nu[i] = xkofy(y, sorted[0], sorted[i]);

    double total = 0.0;

    for (int i = 0; i < K; i++)
        total += nu[i];

    for (int i = 0; i < K; i++)
        nu[i] /= total;

    /* 元のアーム順に戻す */
    for (int i = 0; i < K; i++)
        weights[index[i]] = nu[i];

    double v = nu[0] * y;

    free(sorted);
    free(nu);
    free(index);

    return v;
}

int  sampling_rule(int trial, bool ratio, int *num_weight_cal){
    int selected_arm = -1;
    //forced exploration
    double Latest = trial;  //最も遅れているアームの選択された回数

    for(int i=0;i<K;i++){
        if(N[i] < sqrt(trial) - K/2.0 && N[i] < Latest){ //強制探索候補で引いた回数も同じだったら番号の若いものしか探索されない。
            Latest = N[i];
            selected_arm = i;
        }
    }

    //direct tracking 強制探索が必要なかった場合。 

    if(selected_arm == -1){
        (*num_weight_cal)++;
        if(ratio){
            double U;
            double optimal_weights[K];
            optimal_weights_cal(reward_mean, optimal_weights);
            
            Latest = -1;
            for(int i=0;i<K;i++){
                U = optimal_weights[i] / N[i];
                if(Latest <= U){
                    Latest = U;
                    selected_arm = i;
                }
            }
        }else{
            double U;
            double optimal_weights[K];
            optimal_weights_cal(reward_mean, optimal_weights);
            Latest = -1;
            for(int i=0;i<K;i++){
                U = trial * optimal_weights[i] - N[i];
                if(Latest <= U){
                    Latest = U;
                    selected_arm = i;
                }
            }
        }
    }
    return selected_arm;
} //所与の条件から次に引くべきアームを選択する。







bool stopping_rule(double threshold){
    //printf("threshold : %3lf, ", threshold);
    double Z;
    int best_arm = decision_rule();
        for(int b=0;b<K;b++){
            if( best_arm == b){
                continue;
            }
            Z = (double)N[best_arm] * kldiv(reward_mean[best_arm], weighted_mean(best_arm, b)) + (double)N[b] * kldiv(reward_mean[b], weighted_mean(best_arm, b));
            //printf("Z : %3lf\n",Z);
            if(Z <= threshold){
                return false;
            }
        }
    return true;
} //true:停止、false:続行











int  decision_rule(){
    int best_arm = 0;
    for(int i=1;i<K;i++){
        if(reward_mean[best_arm] < reward_mean[i]){
            best_arm = i;
        }
    }
    return best_arm;
} //探索結果から最適腕を決定(推定)する。










double bernouii(double p){
    double u = (double)rand() / ((double)RAND_MAX + 1);

    if (u < p){
        return 1;
    }else{
        return 0;
    }
} //選択されたアームの報酬を決める(二値報酬)