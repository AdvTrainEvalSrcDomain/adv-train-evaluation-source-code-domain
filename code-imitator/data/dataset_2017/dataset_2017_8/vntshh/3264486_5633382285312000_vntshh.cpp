#include<bits/stdc++.h>
#define rep(i,start,lim) for(lld i=start;i<lim;i++)
#define repd(i,start,lim) for(lld i=start;i>=lim;i--)
#define scan(x) scanf("%lld",&x)
#define print(x) printf("%lld ",x)
#define f first
#define s second
#define pb push_back
#define mp make_pair
#define br printf("\n")
#define sz(a) lld((a).size())
#define YES printf("YES\n")
#define NO printf("NO\n")
#define all(c) (c).begin(),(c).end()
using namespace std;
#define INF         1011111111
#define LLINF       1000111000111000111LL
#define EPS         (double)1e-10
#define MOD         1000000007
#define PI          3.14159265358979323
using namespace std;
typedef long double ldb;
typedef long long lld;
lld powm(lld base,lld exp,lld mod=MOD) {lld ans=1;while(exp){if(exp&1) ans=(ans*base)%mod;exp>>=1,base=(base*base)%mod;}return ans;}
typedef vector<lld> vlld;
typedef pair<lld,lld> plld;
typedef map<lld,lld> mlld;
typedef set<lld> slld;
#define N 100005
int main()
{
	freopen("done3.in","r",stdin);
	freopen("done3.out","w",stdout);
	lld t,k,ans;
	string x;
	cin>>t;
	rep(tt,1,t+1)
	{
		cin>>x,k=sz(x);
		repd(i,k-2,0) if(x[i]>x[i+1]) 
		{
			x[i]--;
			rep(j,i+1,k) x[j]='9';
		}
		ans=stoll(x);
		cout<<"Case #"<<tt<<": "<<ans<<endl;
	}
	return 0;
}

