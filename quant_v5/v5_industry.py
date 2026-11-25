# -*- coding: utf-8 -*-
"""v5_industry.py —— 行业分类（东财口径 → 归一化到一级）。

问题：东财板块是混层级的（"电子"和"数字芯片设计"同时存在），
直接拿来算行业中性会把同一板块拆开，行业偏离约束就失真了。

做法：先拉全部板块成分（v5_industry.build 已有 meta/industry.csv），
再用关键词规则把 190 个板块**归一化**成约 30 个一级行业。
规则按"越具体越先匹配"排序。

输出：meta/industry_coarse.csv（code,name,industry,industry_coarse）
"""
from __future__ import annotations

import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v5_data as D  # noqa: E402

# (关键词列表, 一级行业) —— 顺序敏感：越具体的越靠前
RULES: list[tuple[tuple[str, ...], str]] = [
    (("半导体", "芯片", "集成电路", "分立器件", "光学光电子", "LED", "面板", "元件", "PCB", "电子",
      "消费电子", "被动元件", "电子化学品", "印制电路"), "电子"),
    (("软件", "计算机", "IT服务", "信息技术", "云服务", "算力"), "计算机"),
    (("医药", "医疗", "中药", "化学制药", "生物制品", "原料药", "体外诊断", "诊断", "疫苗", "药"), "医药生物"),
    (("银行", "农商行", "城商行", "村镇银行"), "银行"),
    (("保险", "证券", "多元金融", "金融服务", "非银", "信托", "期货", "租赁", "金控"), "非银金融"),
    (("房地产", "住宅开发", "商业地产", "产业地产", "房产"), "房地产"),
    (("白酒", "啤酒", "饮料", "乳品", "食品", "调味", "预制菜", "零食"), "食品饮料"),
    (("家电", "家用电器", "厨卫"), "家用电器"),
    (("电池", "锂电", "光伏", "风电", "电网", "储能", "电机", "电力设备", "电源设备",
      "电气设备", "输变电", "配电"), "电力设备"),
    (("汽车", "乘用车", "商用车", "零部件", "车身", "底盘", "轮胎"), "汽车"),
    (("军工", "国防", "航空装备", "航天装备", "兵器", "船舶"), "国防军工"),
    (("机械设备", "通用设备", "专用设备", "自动化设备", "工程机械", "机床", "仪器仪表",
      "金属制品", "机器人"), "机械设备"),
    (("化工", "化学", "塑料", "橡胶", "化纤", "农药", "化肥", "氟化工", "磷化工", "钛白粉",
      "有机硅", "民爆", "涂料", "聚氨酯", "聚酯", "涤纶", "粘胶"), "基础化工"),
    (("有色", "黄金", "稀土", "小金属", "工业金属", "能源金属", "铝", "铜", "钨", "钼", "磁性材料"), "有色金属"),
    (("钢铁", "特钢"), "钢铁"),
    (("煤炭", "焦炭", "动力煤"), "煤炭"),
    (("石油", "石化", "油服", "炼化"), "石油石化"),
    (("建筑", "工程咨询", "装修", "基建", "钢结构", "专业工程"), "建筑装饰"),
    (("建材", "水泥", "玻璃", "玻纤", "管材"), "建筑材料"),
    (("电力", "水务", "燃气", "环保", "公用事业", "供热"), "公用事业"),
    (("运输", "港口", "航运", "航空机场", "物流", "公路", "铁路", "快递"), "交通运输"),
    (("通信", "运营商", "光模块", "光纤"), "通信"),
    (("传媒", "游戏", "影视", "广告", "出版", "教育", "院线", "数字媒体"), "传媒"),
    (("零售", "商贸", "贸易", "电商", "百货", "超市", "供应链", "批发"), "商贸零售"),
    (("旅游", "酒店", "餐饮", "景区", "社会服务", "免税", "专业服务", "人力资源", "检测"), "社会服务"),
    (("纺织", "服装", "家纺", "服饰"), "纺织服饰"),
    (("轻工", "造纸", "用纸", "包装", "家居", "文娱用品"), "轻工制造"),
    (("农业", "农林", "牧渔", "种植", "养殖", "饲料", "渔业", "林业", "动保", "农产品", "种子"), "农林牧渔"),
    (("美容", "护理", "化妆品"), "美容护理"),
    (("综合",), "综合"),
]


def _coarse(board: str) -> str:
    for kws, name in RULES:
        if any(k in board for k in kws):
            return name
    return "其他"


def build_coarse(refresh: bool = False) -> pd.DataFrame:
    src = os.path.join(D.C.META, "industry.csv")
    dst = os.path.join(D.C.META, "industry_coarse.csv")
    if os.path.exists(dst) and not refresh:
        return pd.read_csv(dst, dtype={"code": str})
    if not os.path.exists(src):
        raise SystemExit("缺 meta/industry.csv，先跑 v5_industry.py")
    ind = pd.read_csv(src, dtype={"code": str})
    ind["industry_coarse"] = ind["industry"].map(_coarse)
    ind.to_csv(dst, index=False, encoding="utf-8")
    print(f"[coarse] {ind['industry'].nunique()} 个板块 → {ind['industry_coarse'].nunique()} 个一级行业", flush=True)
    return ind


def coverage() -> None:
    uni = D.universe()
    ind = build_coarse()
    m = uni.merge(ind[["code", "industry", "industry_coarse"]], on="code", how="left")
    miss = m[m["industry_coarse"].isna()]
    other = m[m["industry_coarse"] == "其他"]
    print(f"[coverage] 池内 {len(uni)} 只｜有标签 {len(uni)-len(miss)}｜缺 {len(miss)}｜归到'其他' {len(other)}", flush=True)
    if len(miss):
        print("  缺：" + ",".join(miss["code"].head(8) + miss["name"].head(8)), flush=True)
    if len(other):
        print("  '其他'样例（板块）：" + ", ".join(sorted(other["industry"].dropna().unique())[:15]), flush=True)
    vc = m["industry_coarse"].value_counts()
    print(f"[coverage] 共 {len(vc)} 个一级行业，最大 {vc.iloc[0]}（{vc.index[0]}），最小 {vc.iloc[-1]}（{vc.index[-1]}）", flush=True)
    print(vc.to_string(), flush=True)


if __name__ == "__main__":
    if "--coverage" in sys.argv:
        coverage()
    else:
        build_coarse(refresh=True)
        coverage()
