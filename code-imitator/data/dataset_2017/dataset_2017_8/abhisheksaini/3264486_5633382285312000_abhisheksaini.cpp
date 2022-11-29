#include <bits/stdc++.h>
#include <stdlib.h>
#include <time.h> 

using namespace std;
long long mod=1e9+7;
long long INF=1e7;
double EPS = 1e-12;
typedef long long int lld;
#define rep(i,a,n) for(long long int i = (a); i <= (n); ++i)
#define repI(i,a,n) for(int i = (a); i <= (n); ++i)
#define repD(i,a,n) for(long long int i = (a); i >= (n); --i)
#define repDI(i,a,n) for(int i = (a); i >= (n); --i)
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define sc(a) scanf("%lld",&a)
#define sc2(a,b) scanf("%lld%lld",&a,&b)
#define sc3(a,b,c) scanf("%lld%lld%lld",&a,&b,&c)
#define scd(a) scanf("%d",&a)
#define scd2(a,b) scanf("%d%d",&a,&b)
#define scd3(a,b,c) scanf("%d%d%d",&a,&b,&c)
#define scf(a) scanf("%lf",&a)
#define scf2(a,b) scanf("%lf%lf",&a,&b)
#define scf3(a,b,c) scanf("%lf%lf%lf",&a,&b,&c)
#define prL(a) printf("%lld\n",a)
#define prS(a) printf("%lld ",a)
#define prdL(a) printf("%d\n",a)
#define prdS(a) printf("%d ",a)
#define all(c) c.begin(), c.end()
#define sz(a) ((int)a.size())
#define Error(x) cout<< #x << " = " << (x) <<endl;
#define Error2(a,b) cout<<"( "<<#a<<" , "<<#b<<" ) = ( "<<(a)<<" , "<<(b)<<" )\n";
#define Error3(a,b,c) cout<<"("<<#a<<" , "<<#b<<" , "<<#c<<" ) = ( "<<(a)<<" , "<<(b)<<" , "<<(c)<<")\n";
#define Error4(a,b,c,d) cout<<"("<<#a<<" , "<<#b<<" , "<<#c<<" , "<<#d<<" ) = ( "<<(a)<<" , "<<(b)<<" , "<<(c)<<" , "<<(d)<<")\n";
#define errop(a) cout<<#a<<" = ( "<<((a).x)<<" , "<<((a).y)<<" )\n";
#define mset(a, v) memset(a, v, sizeof(a))
#define popcount __builtin_popcountll
typedef pair<lld,lld> PA;

lld powP(lld a,lld b){
    if(b==0) return 1%mod;
    lld k;
    k=powP(a,b/2);
    k=k*k%mod;
    if(b%2==0) return k;
    else return a*k%mod;
}

bool bitSet(lld n,lld i){
    if((n&(1LL<<i))!=0) return true;
    else return false;
}

lld findGcd(lld a,lld b){
    if(a==0) return b;
    else return findGcd(b%a,a);
}

#define lim 700005
#define lim2 200003
#define qqqqqq 210
// inline lld sqr(lld x) { return x * x; }
// unordered_map<lld,lld>::iterator it;
// std::ios::sync_with_stdio(false);
// priority_queue<PA> Q;
// lld dp[1<<18];
// vector<lld> V[lim];
// string S[lim];
lld A[23][12],B[lim],C[lim];
// unordered_map<lld,lld> M,X;
// map<lld,lld> M;

lld countDig(lld n){
    lld ans = 0;
    while(n>0){
        ans++;
        n /=10;
    }
    return ans;
}

int main(){
    lld T,i,n,m,p,j,l,e,r,b,c,k,q,a,d,w,x,y,v,z,t,curr,prev,sum,ans,pos,val,countA,secondMin,indicator;
    rep(i,1,9) A[1][i] = i;
    rep(i,2,18) rep(j,1,9) A[i][j] = A[i-1][j]*10+j;
    B[0] = 1;
    rep(i,1,18) B[i] = B[i-1]*10;
    A[19][1] = A[18][1]*10+1;
    sc(T);
    rep(testC,1,T){
        printf("Case #%lld: ",testC );
        sc(n);
        t = countDig(n);
        if(n<A[t][1]) prL(A[t-1][9]);
        else{
            curr = 0;
            a = 1;
            bool flag;
            repD(j,t,1){
                flag = false;
                rep(i,a,9){
                    val = curr+A[j][i];
                    if(val>n){
                        curr = curr+(i-1)*B[j-1];
                        a = i-1;
                        flag = true;
                        break;
                    }
                }
                if(!flag){
                    curr = curr+9*B[j-1];
                    a = 9;
                }
            }
            prL(curr);
        }
    }
    return 0;
}