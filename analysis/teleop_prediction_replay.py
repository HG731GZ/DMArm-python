"""离线辨识从端自由空间模型，回放原始阻抗与预测残差反馈。"""

from pathlib import Path
import ast
import csv
import json

import numpy as np
from numba import njit
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs" / "teleop_20261124_114847_442757.csv"
OUTPUT = ROOT / "logs" / "offline_replay_20261124"
MODEL_STEP_S = 0.0025  # 自由空间预测器积分步长，单位 s。
IDENTIFY_STEP_S = 0.005  # 连续区间的加速度辨识重采样步长，单位 s。
ACCEL_WINDOW = 7  # 离线二阶平滑窗口为 7 点，共 35 ms；两种来源使用同一窗口。
GAP_LIMIT_S = 0.030  # 加速度辨识不跨越大于 30 ms 的回包间隔。
SPEED_LIMITS = np.array([.734, .773, .755, .925, .889, .885])  # 前次分析得到的平台速度，单位 rad/s。
WINDOWS = ((4.5,12.),(17.8,28.5),(29.2,38.4),(38.6,44.9),(45.,52.1),(52.6,59.3))
SPLITS = (8.1,23.1,33.8,41.7,48.6,57.5)  # 仅用于滞后参数前组标定、后组核对。
METHODS = ("native", "velocity_diff")


def feedback_settings():
    """静态读取现用参数，不导入界面或启动设备。"""
    names = ("FORCE_FEEDBACK_STIFFNESS_NM_PER_RAD", "FORCE_FEEDBACK_DAMPING_NM_S_PER_RAD",
             "FORCE_FEEDBACK_LIMIT_NM", "FORCE_FEEDBACK_VELOCITY_FILTER_S")
    values = {}
    for node in ast.parse((ROOT/"main.py").read_text(encoding="utf-8-sig")).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
            if name in names:
                values[name] = ast.literal_eval(node.value)
    return dict(zip(("stiffness","damping","torque_limit","velocity_filter_s"),[values[x] for x in names]))


def load_data():
    """同时保留日志时刻的缓存值和按接收时刻去重的从端样本。"""
    raw=np.genfromtxt(SOURCE,delimiter=",",names=True,encoding="utf-8-sig",
                      missing_values="NULL",filling_values=np.nan)
    t=raw["elapsed_s"]
    received=raw["slave_received_unix_s"]
    rx=t-(raw["time_unix_s"]-received)
    unique=np.r_[True,np.diff(received)!=0]
    def group(side,field):
        return np.column_stack([raw[f"{side}_j{i}_{field}"] for i in range(1,7)])
    data={"t":t,"rx":rx,"unique":unique,"raw":raw}
    for side,fields in (("master",("q_rad","dq_rad_s","filtered_velocity_error_rad_s")),
                        ("slave",("q_rad","dq_rad_s","ddq_rad_s2"))):
        for field in fields:
            x=group(side,field)
            data[side+"_"+field]=np.unwrap(x,axis=0) if field=="q_rad" else x
    return data


def crossing(t,v,left,right,level):
    """定位相邻样本间的阈值跨越时刻。"""
    return float(t[left]+(t[right]-t[left])*(level-v[left])/(v[right]-v[left]))


def transition_windows(data):
    """仅根据速度选取 20%～80% 的中段，选择过程不读取原生加速度。"""
    t=data["rx"][data["unique"]]
    velocity=data["slave_dq_rad_s"][data["unique"]]
    events=[]
    for axis,speed in enumerate(SPEED_LIMITS):
        for direction in (1,-1):
            v=direction*velocity[:,axis]
            indices=np.flatnonzero(v>.8*speed)
            groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            for group in groups:
                if not len(group) or t[group[-1]]-t[group[0]]<.12:
                    continue
                left,right=int(group[0]),int(group[-1])
                if left==0 or right==len(t)-1:
                    continue
                before=np.flatnonzero((t<t[left])&(t>t[left]-.7)&(v<.2*speed))
                after=np.flatnonzero((t>t[right])&(t<t[right]+.7)&(v<.2*speed))
                for kind,low in (("accel",before[-1] if len(before) else -1),
                                 ("brake",after[0] if len(after) else -1)):
                    if low<0:
                        continue
                    if kind=="accel":
                        begin=crossing(t,v,low,low+1,.2*speed)
                        end=crossing(t,v,left-1,left,.8*speed)
                    else:
                        begin=crossing(t,v,right,right+1,.8*speed)
                        end=crossing(t,v,low-1,low,.2*speed)
                    # 过短的突跳无法辨识斜率，长段已超出单次加减速中段。
                    if .020<=end-begin<=.35:
                        events.append(dict(axis=axis,direction=direction,kind=kind,start=begin,end=end))
    # J5 主激励的制动段缺测，补充同一日志中两段连续可见的制动。
    for direction,left,right in ((1,43.52,43.66),(-1,57.87,58.04)):
        v=direction*velocity[:,4]
        bounds=[]
        for fraction in (.8,.2):
            k=np.flatnonzero((t[:-1]>=left)&(t[1:]<=right)&
                             (v[:-1]>fraction*SPEED_LIMITS[4])&(v[1:]<=fraction*SPEED_LIMITS[4]))[0]
            bounds.append(crossing(t,v,k,k+1,fraction*SPEED_LIMITS[4]))
        events.append(dict(axis=4,direction=direction,kind="brake",start=bounds[0],end=bounds[1]))
    return events


def identify_acceleration(data):
    """两条路径共用时间、速度和事件选择，各自独立计算加减速度幅值。"""
    times=data["rx"][data["unique"]]
    grid=np.arange(0,data["t"][-1],IDENTIFY_STEP_S)
    velocity=data["slave_dq_rad_s"][data["unique"]]
    native=data["slave_ddq_rad_s2"][data["unique"]]
    acceleration={method:np.full((len(grid),6),np.nan) for method in METHODS}
    valid=np.zeros(len(grid),dtype=bool)
    segments=np.split(np.arange(len(times)),np.flatnonzero(np.diff(times)>GAP_LIMIT_S)+1)
    for segment in segments:
        selected=(grid>=times[segment[0]])&(grid<=times[segment[-1]])
        if sum(selected)<ACCEL_WINDOW:
            continue
        local_grid=grid[selected]
        local_velocity=np.column_stack([np.interp(local_grid,times[segment],velocity[segment,i]) for i in range(6)])
        local_native=np.column_stack([np.interp(local_grid,times[segment],native[segment,i]) for i in range(6)])
        # 速度差分分支只使用速度；各连续段独立平滑，端点按段内多项式处理。
        difference=np.gradient(local_velocity,IDENTIFY_STEP_S,axis=0)
        acceleration["native"][selected]=savgol_filter(local_native,ACCEL_WINDOW,2,axis=0)
        acceleration["velocity_diff"][selected]=savgol_filter(difference,ACCEL_WINDOW,2,axis=0)
        valid[selected]=True
    events=[]
    for event in transition_windows(data):
        axis=event["axis"];begin=event["start"];end=event["end"]
        lo=max(0,np.searchsorted(times,begin)-1)
        hi=min(len(times),np.searchsorted(times,end)+1)
        selected=(grid>=begin)&(grid<=end)
        if np.max(np.diff(times[lo:hi]))>GAP_LIMIT_S or not valid[selected].all():
            continue
        points=np.r_[begin,grid[selected],end]
        factor=event["direction"]*(1 if event["kind"]=="accel" else -1)
        for method in METHODS:
            # 只在该连续段的有效网格上插值，避免段边缘的空值混入事件均值。
            local=(grid>=begin-IDENTIFY_STEP_S)&(grid<=end+IDENTIFY_STEP_S)&valid
            values=np.interp(points,grid[local],acceleration[method][local,axis])
            event[method]=float(factor*np.trapezoid(values,points)/(end-begin))
        event["central_slope"]=float(.6*SPEED_LIMITS[axis]/(end-begin))
        event["primary"]=bool(WINDOWS[axis][0]<=begin<=WINDOWS[axis][1])
        events.append(event)
    params={method:[] for method in METHODS}
    selected_events=[]
    for axis in range(6):
        categories=(("accel_positive","accel",1),("accel_negative","accel",-1),("brake","brake",0))
        chosen={}
        for label,kind,direction in categories:
            candidates=[e for e in events if e["axis"]==axis and e["kind"]==kind and (direction==0 or e["direction"]==direction)]
            primary=[e for e in candidates if e["primary"]]
            chosen[label]=primary if primary else candidates
            if not chosen[label]:
                raise ValueError(f"J{axis+1} 的 {label} 没有连续辨识片段")
            selected_events.extend(dict(e,parameter=label) for e in chosen[label])
        for method in METHODS:
            params[method].append(dict(speed=float(SPEED_LIMITS[axis]),**{
                label:float(np.median([e[method] for e in group])) for label,group in chosen.items()}))
    return params,selected_events,grid,acceleration,valid


@njit
def simulate(command,delay,vmax,acc_pos,acc_neg,brake,q0,v0,step):
    """仅由历史主端位置推进，速度包络预留制动距离，加速和制动分别限幅。"""
    q=np.empty(len(command));v=np.empty(len(command))
    q[0]=q0;v[0]=v0
    for k in range(len(command)-1):
        past=max(0.,k-delay/step)
        left=int(past);fraction=past-left
        target=command[left]*(1-fraction)+command[min(left+1,k)]*fraction
        error=target-q[k]
        sign=1. if error>=0 else -1.
        # 按当前位置误差收紧目标速度，使停靠过程遵守制动减速度。
        goal=sign*min(vmax,np.sqrt(2*brake*abs(error)))
        change=goal-v[k]
        speeding=v[k]*change>=0
        if speeding:
            rate=acc_pos if goal>=0 else acc_neg
        else:
            rate=brake
        change=min(rate*step,max(-rate*step,change))
        next_v=v[k]+change
        # 反向时先完成制动，下一小步再按另一方向的加速度起步。
        if v[k]*next_v<0 and not speeding:
            next_v=0.
        v[k+1]=next_v
        q[k+1]=q[k]+.5*(v[k]+next_v)*step
    return q,v


def fit_models(data,params):
    """固定速度、加速和制动参数，仅重新拟合每轴一个滞后参数。"""
    grid=np.arange(0,data["t"][-1]+MODEL_STEP_S,MODEL_STEP_S)
    command_rows=np.clip(np.searchsorted(data["t"],grid,side="right")-1,0,len(data["t"])-1)
    times=data["rx"][data["unique"]]
    q_actual=data["slave_q_rad"][data["unique"]]
    results={};predictions={}
    for method in METHODS:
        results[method]=[];predictions[method]=[]
        for axis,p in enumerate(params[method]):
            command=data["master_q_rad"][command_rows,axis]
            q0=data["slave_q_rad"][0,axis];v0=data["slave_dq_rad_s"][0,axis]
            selected=(times>=WINDOWS[axis][0])&(times<=WINDOWS[axis][1])
            stamps=times[selected];actual=q_actual[selected,axis]
            train=stamps<SPLITS[axis];late=~train
            delays=np.arange(0,.201,.001)
            scores=[];early_scores=[];late_scores=[]
            for delay in delays:
                q,v=simulate(command,delay,p["speed"],p["accel_positive"],p["accel_negative"],p["brake"],q0,v0,MODEL_STEP_S)
                error=np.interp(stamps,grid,q)-actual
                scores.append(np.mean(error**2))
                early_scores.append(np.mean(error[train]**2))
                late_scores.append(np.mean(error[late]**2))
            best=int(np.argmin(scores));early_best=int(np.argmin(early_scores))
            q,v=simulate(command,delays[best],p["speed"],p["accel_positive"],p["accel_negative"],p["brake"],q0,v0,MODEL_STEP_S)
            # 与从端缓存帧的接收时刻配对；重复旧帧时预测基线也保持在同一时刻。
            pred_q=np.interp(data["rx"],grid,q)
            pred_v=np.interp(data["rx"],grid,v)
            early_q,early_v=simulate(command,delays[early_best],p["speed"],p["accel_positive"],p["accel_negative"],p["brake"],q0,v0,MODEL_STEP_S)
            results[method].append(dict(p,delay_s=float(delays[best]),
                fit_position_rmse_deg=float(np.rad2deg(np.sqrt(scores[best]))),
                early_delay_s=float(delays[early_best]),
                late_position_rmse_deg=float(np.rad2deg(np.sqrt(late_scores[early_best])))))
            predictions[method].append((pred_q,pred_v,q,v,
                np.interp(data["rx"],grid,early_q),np.interp(data["rx"],grid,early_v)))
            print(method,"J",axis+1,"limits,delay_ms,fit_rmse_deg,late_rmse_deg",[round(x,4) for x in (
                p["accel_positive"],p["accel_negative"],p["brake"],1000*delays[best],
                np.rad2deg(np.sqrt(scores[best])),np.rad2deg(np.sqrt(late_scores[early_best])))] ,flush=True)
    return results,predictions,grid


@njit
def lowpass(t,values,tau):
    """采用控制器相同的一阶指数滤波，按日志实际时间间隔更新。"""
    output=np.empty_like(values);output[0]=values[0]
    for k in range(1,len(t)):
        alpha=1. if tau==0 else -np.expm1(-(t[k]-t[k-1])/tau)
        output[k]=output[k-1]+alpha*(values[k]-output[k-1])
    return output


def wrapped(x):
    """把关节角差映射到最短角差范围。"""
    return (x+np.pi)%(2*np.pi)-np.pi


def torque_replay(data,predictions,settings):
    """在相同主端运动轨迹上计算关节侧反馈，统一使用现用合成限幅。"""
    k=np.asarray(settings["stiffness"]);b=np.asarray(settings["damping"]);limit=np.asarray(settings["torque_limit"])
    tau=settings["velocity_filter_s"]
    original=k*wrapped(data["slave_q_rad"]-data["master_q_rad"])+b*data["master_filtered_velocity_error_rad_s"]
    torque={"original":np.clip(original,-limit,limit)}
    raw_torque={"original":original}
    qpred={};vpred={}
    for method in METHODS:
        qpred[method]=np.column_stack([x[0] for x in predictions[method]])
        vpred[method]=np.column_stack([x[1] for x in predictions[method]])
        velocity_error=lowpass(data["t"],data["slave_dq_rad_s"]-vpred[method],tau)
        result=k*wrapped(data["slave_q_rad"]-qpred[method])+b*velocity_error
        raw_torque[method]=result;torque[method]=np.clip(result,-limit,limit)
    reconstructed=lowpass(data["t"],data["slave_dq_rad_s"]-data["master_dq_rad_s"],tau)
    discrepancy=b*(reconstructed-data["master_filtered_velocity_error_rad_s"])
    print("原始阻尼滤波按日志重建与记录值的力矩 RMS 差异",np.sqrt(np.mean(discrepancy**2,axis=0)),flush=True)
    return torque,raw_torque,qpred,vpred


def summarize(data,torque,settings):
    """用时间权重计算幅值和阻力做功，保持短时回包间断在全日志指标中。"""
    t=data["t"];dt=np.r_[np.diff(t),0]
    v=data["master_dq_rad_s"]
    rows=[]
    limits=np.asarray(settings["torque_limit"])
    for scope in ("full_log","primary_window","later_window","fresh_rows"):
        for axis in range(6):
            if scope=="full_log":
                mask=np.ones(len(t),dtype=bool)
            elif scope=="fresh_rows":
                mask=t-data["rx"]<=.020
            else:
                start=WINDOWS[axis][0] if scope=="primary_window" else SPLITS[axis]
                mask=(t>=start)&(t<=WINDOWS[axis][1])
            weight=dt*mask
            moving=abs(v[:,axis])>.05
            for method,values in torque.items():
                value=values[:,axis]
                work=float(np.sum(np.maximum(0,-value*v[:,axis])*weight))
                aid=float(np.sum(np.maximum(0,value*v[:,axis])*weight))
                moving_weight=weight*moving
                opposing=np.maximum(0,-value*np.sign(v[:,axis]))
                rows.append(dict(scope=scope,axis=axis+1,method=method,
                    rms_nm=float(np.sqrt(np.sum(value**2*weight)/sum(weight))),
                    p95_abs_nm=float(np.quantile(abs(value[mask]),.95)),
                    peak_abs_nm=float(np.max(abs(value[mask]))),
                    resisting_work_j=work,assisting_work_j=aid,
                    moving_mean_resisting_nm=float(np.sum(opposing*moving_weight)/sum(moving_weight)),
                    saturated_fraction=float(np.sum((abs(value)>=limits[axis]-1e-9)*weight)/sum(weight))))
    return rows


def write_csv(path,rows):
    """按记录字段导出可直接读取的数值表。"""
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def save_replay(data,torque,qpred,vpred):
    """导出逐样本回放，保留全部真实日志采样时刻。"""
    arrays=[data["t"],data["rx"]];headers=["elapsed_s","slave_receive_elapsed_s"]
    groups={"master_q_rad":data["master_q_rad"],"master_dq_rad_s":data["master_dq_rad_s"],
            "slave_q_rad":data["slave_q_rad"],"slave_dq_rad_s":data["slave_dq_rad_s"]}
    for method in METHODS:
        groups[method+"_predicted_q_rad"]=qpred[method]
        groups[method+"_predicted_dq_rad_s"]=vpred[method]
    for method,values in torque.items():
        groups[method+"_feedback_nm"]=values
    for label,values in groups.items():
        for axis in range(6):
            headers.append(f"j{axis+1}_{label}");arrays.append(values[:,axis])
    np.savetxt(OUTPUT/"replay_samples.csv",np.column_stack(arrays),delimiter=",",
               header=",".join(headers),comments="",fmt="%.9g",encoding="utf-8-sig")


def plot_results(data,torque,results,metrics):
    """绘制反馈回放与方法对比，供离线人工预览。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,
                        "font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={"original":"#9a5a47","native":"#087f8c","velocity_diff":"#8d50ae"}
    labels={"original":"原始阻抗反馈","native":"预测补偿 · 回传加速度","velocity_diff":"预测补偿 · 速度差分"}
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    t=data["t"]
    for axis,panel in enumerate(axes.flat):
        start,end=WINDOWS[axis];mask=(t>=start)&(t<=end)
        for method in ("original",*METHODS):
            panel.plot(t[mask],torque[method][mask,axis],color=colors[method],
                       linewidth=1.1 if method=="original" else 1.35,
                       linestyle="--" if method=="velocity_diff" else "-",alpha=.9)
        panel.axhline(0,color="#bcc4ca",linewidth=.6)
        limit=5.5 if axis<3 else 2.2
        panel.set(xlim=(start,end),ylim=(-limit,limit),xlabel="日志经过时间（s）",ylabel="关节侧反馈力矩（N·m）")
        lookup={r["method"]:r for r in metrics if r["axis"]==axis+1 and r["scope"]=="primary_window"}
        reduction=100*(1-lookup["native"]["rms_nm"]/lookup["original"]["rms_nm"])
        panel.set_title(f"J{axis+1}    主要激励段 RMS 降低 {reduction:.1f}%",loc="left",fontsize=12)
        panel.grid(axis="y",color="#e6ebee",linewidth=.6)
    fig.suptitle("同一无接触轨迹：预测补偿前后的反馈力矩",x=.07,ha="left",y=.98,fontsize=19)
    fig.legend(handles=[Line2D([0],[0],color=colors[m],ls="--" if m=="velocity_diff" else "-",label=labels[m])
                        for m in ("original",*METHODS)],loc="upper left",bbox_to_anchor=(.065,.945),ncol=3,frameon=False)
    fig.text(.07,.012,"保持 main.py 的 K、B 与合成力矩限幅；预测基线按从端帧接收时刻配对；包含原日志的回包间断。",fontsize=9,color="#53636c")
    fig.tight_layout(rect=(0,.035,1,.90),h_pad=2,w_pad=2)
    fig.savefig(OUTPUT/"feedback_comparison.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for panel,field,title,unit in zip(axes,("rms_nm","resisting_work_j"),
                                    ("全日志反馈力矩 RMS","全日志沿运动反向的阻力做功"),("N·m","J")):
        x=np.arange(6)
        for index,method in enumerate(("original",*METHODS)):
            y=[r[field] for r in metrics if r["method"]==method and r["scope"]=="full_log"]
            panel.bar(x+(index-1)*.24,y,width=.23,color=colors[method],label=labels[method])
        panel.set_xticks(x,[f"J{i}" for i in range(1,7)])
        panel.set_ylabel(unit);panel.set_title(title,loc="left",fontsize=13)
        panel.grid(axis="y",color="#e6ebee",linewidth=.6);panel.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper left",bbox_to_anchor=(.065,.97),ncol=3,frameon=False)
    fig.text(.07,.015,"做功按固定日志轨迹计算：∫max(0, −反馈力矩 × 主端角速度) dt。",fontsize=10,color="#53636c")
    fig.tight_layout(rect=(0,.055,1,.87),w_pad=3)
    fig.savefig(OUTPUT/"feedback_metrics.png",dpi=160);plt.close(fig)


def plot_acceleration(data,time,acceleration,valid):
    """对照典型起步和急停的加速度来源，显示实际使用的离线滤波尺度。"""
    import matplotlib.pyplot as plt
    t=data["rx"][data["unique"]]
    raw=data["slave_ddq_rad_s2"][data["unique"]]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for panel,(axis,start,end,title) in zip(axes,((2,29.45,29.9,"J3 正向起步"),(5,58.8,59.08,"J6 正向制动"))):
        selected=(t>=start)&(t<=end)
        panel.plot(t[selected],raw[selected,axis],color="#adb6bf",linewidth=1,label="回传原始值")
        for method,color,label in (("native","#087f8c","回传加速度 · 平滑"),("velocity_diff","#8d50ae","速度差分 · 平滑")):
            selected=(time>=start)&(time<=end)
            values=acceleration[method][selected,axis].copy()
            values[~valid[selected]]=np.nan
            panel.plot(time[selected],values,color=color,label=label,linewidth=1.5,ls="--" if method=="velocity_diff" else "-")
        panel.set(xlim=(start,end),xlabel="日志经过时间（s）",ylabel="加速度（rad/s²）")
        panel.set_title(title,loc="left",fontsize=13);panel.grid(axis="y",color="#e6ebee",linewidth=.6)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper left",bbox_to_anchor=(.065,.99),ncol=3,frameon=False)
    fig.text(.07,.015,"连续段内按 5 ms 重采样；速度中心差分；两种加速度均做 35 ms、二阶 Savitzky–Golay 离线平滑。",fontsize=9,color="#53636c")
    fig.tight_layout(rect=(0,.06,1,.87),w_pad=3)
    fig.savefig(OUTPUT/"acceleration_comparison.png",dpi=160);plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    data=load_data();settings=feedback_settings()
    parameters,events,accel_time,acceleration,valid=identify_acceleration(data)
    write_csv(OUTPUT/"identification_events.csv",events)
    results,predictions,model_time=fit_models(data,parameters)
    write_csv(OUTPUT/"model_parameters.csv",[
        dict(axis=axis+1,acceleration_source=method,
             speed_limit_rad_s=p["speed"],accel_positive_rad_s2=p["accel_positive"],
             accel_negative_rad_s2=p["accel_negative"],braking_rad_s2=p["brake"],delay_s=p["delay_s"])
        for method in METHODS for axis,p in enumerate(results[method])])
    torque,raw_torque,qpred,vpred=torque_replay(data,predictions,settings)
    metrics=summarize(data,torque,settings)
    early_predictions={method:[(x[4],x[5]) for x in predictions[method]] for method in METHODS}
    early_torque,_,_,_=torque_replay(data,early_predictions,settings)
    early_metrics=[r for r in summarize(data,early_torque,settings) if r["scope"]=="later_window"]
    write_csv(OUTPUT/"later_motion_metrics.csv",early_metrics)
    write_csv(OUTPUT/"metrics.csv",metrics)
    save_replay(data,torque,qpred,vpred)
    summary={"说明":"能力参数从全段连续速度过渡估计，两种方法仅加速度来源不同；仅以主端历史位置驱动预测。",
             "source":str(SOURCE),"feedback_settings":settings,"model_step_s":MODEL_STEP_S,
             "acceleration_smoothing_window_s":ACCEL_WINDOW*IDENTIFY_STEP_S,
             "模型":"目标速度按 min(vmax, sqrt(2*d*abs(位置误差))) 生成；正负向起步和制动分别限制加速度。",
             "时序":"滞后以接收时间轴拟合，计算反馈时按每行实际使用的从端帧接收时刻取预测基线。",
             "辨识":"能力参数使用全段；滞后同时记录全主激励段拟合和前组拟合后组位置核对结果。",
             "results":results,
             "原始阻抗公式":"clip(K*wrap(q_slave-q_master)+B*日志内滤波速度差, -limit, +limit)",
             "预测补偿公式":"clip(K*wrap(q_slave-q_pred_at_receive)+B*LP(v_slave-v_pred_at_receive), -limit, +limit)",
             "滤波说明":f"速度中心差分和 35 ms 二阶 Savitzky–Golay 只用于离线辨识常数；在线模型不读取加速度。补偿速度残差使用现用 {settings['velocity_filter_s']*1000:g} ms 因果一阶低通。"}
    dt=np.r_[np.diff(data["t"]),0]
    summary["full_log_aggregate"]={method:{
        "six_axis_rss_rms_nm":float(np.sqrt(np.sum(values**2*dt[:,None])/sum(dt))),
        "resisting_work_j":sum(r["resisting_work_j"] for r in metrics if r["method"]==method and r["scope"]=="full_log"),
        "assisting_work_j":sum(r["assisting_work_j"] for r in metrics if r["method"]==method and r["scope"]=="full_log")}
        for method,values in torque.items()}
    summary["acceleration_source_torque_difference_rms_nm"]=np.sqrt(np.sum((torque["native"]-torque["velocity_diff"])**2*dt[:,None],axis=0)/sum(dt)).tolist()
    (OUTPUT/"parameters.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    plot_results(data,torque,results,metrics)
    plot_acceleration(data,accel_time,acceleration,valid)
    for axis in range(6):
        rows=[r for r in metrics if r["axis"]==axis+1 and r["scope"]=="full_log"]
        print("全日志 J",axis+1,"RMS、阻力做功、峰值",[(r["method"],round(r["rms_nm"],4),round(r["resisting_work_j"],4),round(r["peak_abs_nm"],4)) for r in rows],flush=True)
    for method in METHODS:
        work=sum(r["resisting_work_j"] for r in metrics if r["method"]==method and r["scope"]=="full_log")
        print(method,"六轴总阻力做功",work,flush=True)
    print("两种加速度来源的补偿力矩差 RMS",np.sqrt(np.mean((torque["native"]-torque["velocity_diff"])**2,axis=0)),flush=True)


if __name__=="__main__":
    main()
