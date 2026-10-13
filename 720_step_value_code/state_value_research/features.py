"""Current-observation-only features. Future outcomes are deliberately not inputs."""
from collections import Counter, defaultdict
import copy
import math
from mechanics import E

ITEMS = tuple(E.PRODUCTS) + tuple(E.ANIMALS)
STEPS = tuple(d * 24 + h for d in (26, 27, 28, 29) for h in (0, 8, 16))


def shed_distance(pos, n):
    return min(abs(pos[0]-x)+abs(pos[1]-y) for x,y in E._shed_access_tiles(n))


def state_features(obs, cfg):
    me = obs['player']; day = obs['day']; hour = obs['hour']
    last = cfg.get('episodeSteps', 720) - 1
    turn = day * cfg.get('turnsPerDay', 24) + hour
    own = obs['farms'][me]['money']; opp = obs['farms'][1-me]['money']
    f = {'cash.own':float(own), 'cash.opp':float(opp), 'cash.diff':float(own-opp),
         'cash.time_left':float(last-turn), 'cash.hour':float(hour), 'cash.day':float(day)}
    for role, farm in [('own', obs['farms'][me]), ('opp', obs['farms'][1-me])]:
        prefix = 'res.'+role+'.'
        n = len(farm['tiles'])
        f[prefix+'land'] = len(farm['unlocked_quadrants'])
        f[prefix+'workers'] = 1+len(farm['hands'])
        f[prefix+'hires_today'] = farm['hires_today']
        positions = [farm['farmer']]+farm['hands']
        f[prefix+'worker_distance'] = sum(shed_distance(p,n) for p in positions)
        for k in ('empty','locked','weed','COOP_empty','PASTURE_empty'):
            f[prefix+k] = 0
        for k in tuple(E.CROPS)+tuple(E.ANIMALS):
            for attr in ('count','yield','ready_yield','age_sum','distance_sum','due_service',
                         'stress','fertilized','care_pending','fertilizer_available',
                         'mature_0','mature_1','mature_2','mature_3plus','expires_24'):
                f[prefix+k+'.'+attr] = 0
        for y,row in enumerate(farm['tiles']):
            for x,tile in enumerate(row):
                if tile is None: f[prefix+'empty'] += 1; continue
                if tile == 'LOCKED': f[prefix+'locked'] += 1; continue
                if tile['kind']=='WEED': f[prefix+'weed'] += 1; continue
                kind = tile.get('crop') or tile.get('animal')
                if not kind:
                    f[prefix+tile['kind']+'_empty'] += 1; continue
                p = prefix+kind+'.'
                is_crop = 'crop' in tile
                definition = E.CROPS[kind] if is_crop else E.ANIMALS[kind]
                age = day-tile['planted_day' if is_crop else 'placed_day']
                until = max(0, definition['first_yield_day']-age)
                f[p+'count'] += 1; f[p+'yield'] += tile['yield_units']
                f[p+'age_sum'] += age; f[p+'distance_sum'] += shed_distance((x,y),n)
                f[p+'ready_yield'] += tile['yield_units'] if until==0 else 0
                f[p+('mature_'+str(until) if until<3 else 'mature_3plus')] += 1
                if is_crop:
                    f[p+'due_service'] += not tile['watered_today']
                    f[p+'stress'] += tile['consecutive_unwatered']
                    f[p+'fertilized'] += tile.get('fertilized_until_day',-1)>=day
                    expiry=tile.get('max_lifespan_step',-1)
                    f[p+'expires_24'] += expiry>=0 and expiry<=turn+24
                else:
                    f[p+'due_service'] += not tile['fed_today']
                    f[p+'stress'] += tile['consecutive_unfed']
                    f[p+'care_pending'] += tile.get('pending_care_bonus',0)
                    f[p+'fertilizer_available'] += tile.get('fertilizer_available',False)
        # Opponent private state is NOT zero: a separate known mask accompanies
        # placeholders, and no hidden observation is consulted.
        f[prefix+'private_known'] = int(role=='own')
        carried=Counter()
        if role=='own':
            for inv in obs['private']['inventories']: carried.update(inv)
        for item in ITEMS:
            f[prefix+'shed.'+item] = obs['private']['shed'].get(item,0) if role=='own' else 0
            f[prefix+'carried.'+item] = carried[item] if role=='own' else 0
        for crop in E.CROPS:
            f[prefix+'seed.'+crop] = obs['private']['seeds'].get(crop,0) if role=='own' else 0
    for item in E.PRODUCTS:
        f['res.market.price.'+item] = obs['market']['prices'][item]
        f['res.market.supply.'+item] = obs['market']['inventory'][item]
    shops=Counter(obs['town']['unlocked_shops'])
    for shop in E.SHOPS: f['res.shop.'+shop] = shops[shop]
    assert all(math.isfinite(v) for v in f.values())
    return f


def reference_schedule(obs, seat, cfg):
    """Existing cohorts only; unlimited parallel workers, fixed tile actions.

    Uses original official action/biology/decay functions. Transport consumes
    time but not shared workers; no affordability/stock-capacity constraints.
    The same tile cannot WATER and HARVEST in the same turn.
    """
    tpd=cfg.get('turnsPerDay',24); start=obs['day']*tpd+obs['hour']
    last=cfg.get('episodeSteps',720)-1
    farm=copy.deepcopy(obs['farms'][seat]); n=len(farm['tiles'])
    deliveries=defaultdict(Counter); feeds=Counter(); services=Counter()
    if seat==obs['player']:
        deliveries[start].update({k:v for k,v in obs['private']['shed'].items() if k in E.PRODUCTS})
        for pos,inv in zip([farm['farmer']]+farm['hands'],obs['private']['inventories']):
            dt=start+shed_distance(pos,n)+1
            deliveries[dt].update({k:v for k,v in inv.items() if k in E.PRODUCTS})
    # Opponent hidden stocks intentionally not included, not treated as known 0.
    for t in range(start,last):
        day=t//tpd
        for y,row in enumerate(farm['tiles']):
            for x,tile in enumerate(row):
                if not isinstance(tile,dict):continue
                cmd=None; feed=False; lag=shed_distance((x,y),n)+1
                if 'crop' in tile:
                    cd=E.CROPS[tile['crop']]; age=day-tile['planted_day']
                    ready=age>=cd['first_yield_day'] and tile['yield_units']>0
                    # Harvest ongoing output promptly, annuals at maximum age;
                    # clip waiting when delivery would otherwise miss terminal.
                    harvest=ready and (cd['ongoing'] or age>=cd['max_yield_day']
                                       or t+tpd+lag>=last)
                    water_bonus=(not cd['ongoing'] and not tile['watered_today']
                                 and (cd['max_yield_day']+1)//2<=age<=cd['max_yield_day']
                                 and t+1+lag<last)
                    if harvest and not water_bonus:cmd=['HARVEST']
                    elif not tile['watered_today']:cmd=['WATER']
                elif 'animal' in tile:
                    if tile['yield_units'] and t+lag<last:cmd=['HARVEST']
                    elif not tile['fed_today']:cmd=['FEED'];feed=True
                    elif not tile['cared_today']:cmd=['CARE']
                    elif tile['fertilizer_available'] and t+lag<last:cmd=['COLLECT_FERTILIZER']
                if cmd:
                    farm['farmer']=[x,y]
                    private={'shed':{},'seeds':{},'inventories':[{'WHEAT':1} if feed else {}]}
                    E._apply_unit_action(farm,private,0,cmd,n,day,tpd,10**9)
                    if feed:feeds[t]+=1
                    else:deliveries[t+lag].update(private['inventories'][0])
                    services[t]+=1
        E._decay_plants(farm,t)
        if (t+1)%tpd==0:
            E._daily_refresh_plants(farm,day,tpd)
            E._daily_refresh_animals(farm,day)
    return deliveries,feeds,services


def price_schedule(obs,cfg,schedules):
    tpd=cfg.get('turnsPerDay',24); start=obs['day']*tpd+obs['hour']
    last=cfg.get('episodeSteps',720)-1
    shops=obs['town']['unlocked_shops']
    # R0 late window must have no remaining unknown shop draws. Fail rather
    # than silently pretend current shops are the full future for earlier days.
    assert len(shops)==E.MAX_SHOP_INSTANCES, 'Earlier-window scenario extension needed'
    market=copy.deepcopy(obs['market']); inv=market['inventory'];params=market.get('params')
    gross=[Counter(),Counter()];cost=[Counter(),Counter()]
    for t in range(start,last):
        # Feed buys at post-buy inventory; both seats quote before either commits.
        needs=[s[1][t] for s in schedules]
        for j in range(max(needs,default=0)):
            price=E.market_price('WHEAT',inv['WHEAT']-1,params)
            count=0
            for seat in (0,1):
                if j<needs[seat]:cost[seat][t]+=price;count+=1
            inv['WHEAT']-=count
        # Explicit common product ordering; same-product simultaneous units.
        for item in E.PRODUCTS:
            quantities=[s[0].get(t,{}).get(item,0) for s in schedules]
            for j in range(max(quantities,default=0)):
                price=E.market_price(item,inv[item],params);count=0
                for seat in (0,1):
                    if j<quantities[seat]:gross[seat][t]+=price;count+=1
                if price>1:inv[item]+=count
        if t%max(1,cfg.get('townShopSellInterval',4))==0:
            for shop in shops:
                for item in E.SHOPS[shop]:inv[item]-=2 if len(E.SHOPS[shop])==1 else 1
        if t%max(1,cfg.get('townCenterSellInterval',24))==0:
            for item in E.TOWN_CENTER_PRODUCTS:inv[item]-=1
    return gross,cost


def income_features(obs,cfg):
    me=obs['player']; start=obs['day']*cfg.get('turnsPerDay',24)+obs['hour']
    last=cfg.get('episodeSteps',720)-1
    schedules=[reference_schedule(obs,s,cfg) for s in (0,1)]
    gross,cost=price_schedule(obs,cfg,schedules);out={}
    for name,horizon in [('24',24),('48',48),('remaining',last-start)]:
        stop=min(last,start+horizon)
        for role,s in [('own',me),('opp',1-me)]:
            p='income.'+name+'.'+role+'.'
            out[p+'gross']=sum(v for t,v in gross[s].items() if t<stop)
            out[p+'feed_cost']=sum(v for t,v in cost[s].items() if t<stop)
            out[p+'net_feed']=out[p+'gross']-out[p+'feed_cost']
            out[p+'service_actions']=sum(v for t,v in schedules[s][2].items() if t<stop)
            for item in E.PRODUCTS:
                out[p+'units.'+item]=sum(v.get(item,0) for t,v in schedules[s][0].items() if start<=t<stop)
        out['income.'+name+'.margin_net_feed']=out['income.'+name+'.own.net_feed']-out['income.'+name+'.opp.net_feed']
    out['income.opp_hidden_stock_known']=0
    return out
