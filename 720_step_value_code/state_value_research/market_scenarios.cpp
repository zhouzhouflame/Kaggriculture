// Existing-cohort market accounting only. Not a strategic simulator.
#include <cmath>
#include <algorithm>
#include <vector>

static double shape(int f,double x,double t){
    x=std::max(0.0,x);
    switch(f){case 1:return x*x;case 2:return std::sqrt(x);case 3:return std::log(1+x);
        case 4:return std::log10(1+x);case 5:{if(t<=0)return x;double u=x/t;return u+8*std::pow(std::max(0.,u-1),2);}default:return x;}
}
static double price(int item,double inv,const double *params){
    const double*p=params+item*7;double v;
    if(inv<p[1])v=p[0]+p[5]*shape((int)p[3],p[1]-inv,p[2]);
    else v=p[0]-p[6]*shape((int)p[4],inv-p[1],p[2]);
    return std::max(1.,std::nearbyint(v));
}
extern "C" void prices(int n,const int*items,const double*inventories,const double*params,double*out){
    for(int i=0;i<n;i++)out[i]=price(items[i],inventories[i],params);
}
extern "C" void market(int start,int last,int scenarios,int products,int known,int wheat,
    int shop_interval,int shop_sell,int town_sell,const double*initial,const double*params,
    const int*goods,const int*feeds,const int*shop_matrix,const int*current_demand,
    const int*future_shops,const int*town_products,double*gross,double*cost){
    // goods indexed [t,seat,product], outputs [scenario,seat,t].
    for(int sc=0;sc<scenarios;sc++){
        std::vector<double>inv(initial,initial+products);
        std::vector<int>demand(current_demand,current_demand+products);int nsh=known;
        for(int t=start;t<last;t++){
            if(t>start && t%shop_interval==0 && nsh<8){
                int shop=future_shops[sc*8+nsh++];
                for(int j=0;j<products;j++)demand[j]+=shop_matrix[shop*products+j];
            }
            int a=feeds[t*2],b=feeds[t*2+1];
            for(int j=0;j<std::max(a,b);j++){
                double p=price(wheat,inv[wheat]-1,params);int count=0;
                if(j<a){cost[(sc*2)*last+t]+=p;count++;}
                if(j<b){cost[(sc*2+1)*last+t]+=p;count++;}inv[wheat]-=count;
            }
            for(int it=0;it<products;it++){
                a=goods[(t*2)*products+it];b=goods[(t*2+1)*products+it];
                for(int j=0;j<std::max(a,b);j++){
                    double p=price(it,inv[it],params);int count=0;
                    if(j<a){gross[(sc*2)*last+t]+=p;count++;}
                    if(j<b){gross[(sc*2+1)*last+t]+=p;count++;}if(p>1)inv[it]+=count;
                }
            }
            if(t%shop_sell==0)for(int it=0;it<products;it++)inv[it]-=demand[it];
            if(t%town_sell==0)for(int it=0;it<products;it++)inv[it]-=town_products[it];
        }
    }
}
