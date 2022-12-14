#define _CRT_SECURE_NO_WARNINGS
#pragma comment(linker,"/STACK:100000000000")
#include <vector>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <queue>
#include <stack>
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <complex>

using namespace std;
#define FR(i,n) for(int (i)=0;(i)<(n);(i)++)
#define FOR(i,c,n) for(int (i)=(c);(i)<(n);(i)++)
#define REP(it,v,cont) for(cont::iterator (it)=(v).begin();(it)!=(v).end();++(it)) 
#define CLR(a,c) memset((a),(c),sizeof (a))
#define ALL(v) (v).begin(),(v).end()
#define SQR(a) ((a)*(a))
typedef long long ll;
typedef unsigned long long  ull;
typedef long double lld;
typedef pair<int,int> PII;
typedef pair<ll,ll> PLL;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<double> VD;

int main(){
	freopen("a.in","r",stdin);
	freopen("a.out","w",stdout);
	int Tcas;cin>>Tcas;
	FR(cas,Tcas){
		printf("Case #%d: ",cas+1);
		string s;
		int k;
		cin>>s>>k;
		int res=0;
		FR(i,s.size()-k+1){
			if(s[i]=='+') continue;
			FOR(j,i,i+k) s[j]=(s[j]=='+'?'-':'+');
			res++;
		}
		bool ok=true;
		FR(i,s.size()) if(s[i]=='-') ok=false;
		if(ok) cout<<res<<endl;
		else cout<<"IMPOSSIBLE"<<endl;
	}
}
	