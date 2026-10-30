"""Generate the Tableau starter workbook tableau/bank_churn_dashboard.twb from the CSVs in reports/tableau/.

Run from the repo root:  python -m src.reporting.tableau_workbook

The workbook is written as plain XML (a .twb is XML) and read by Tableau Desktop / Tableau Public. Each CSV is its own data source,
connected by a path relative to the workbook (../reports/tableau). There are no joins or blends: every sheet uses one CSV, and the
two parameters reach into each CSV through calculated fields. Column names and types come from the CSV files themselves, so the
workbook cannot reference a field that is not there; validate() re-checks this and the internal references.

Money is in "units" and rests on assumptions; drift scenarios are SIMULATED. Both are labelled on the sheets.
"""
import hashlib
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.models.registry import ROOT

CSV_DIR = ROOT / "reports" / "tableau"
OUT_PATH = ROOT / "tableau" / "bank_churn_dashboard.twb"
REL_CSV_DIR = "../reports/tableau"   # relative to the workbook's folder
VERSION = "18.1"
SOURCE_BUILD = "2026.1.0 (20261.26.0211.1127)"   # the build the schema in tableau-document-schemas describes
PARAM_DS = "Parameters"
DASHBOARD = "Bank Churn & Retention"
INK = "#0b0b0b"
GREEN, AMBER, RED, BLUE, GREY = "#1baf7a", "#eb9d34", "#d03b3b", "#2a78d6", "#9a9a95"
DTYPE = {"int64": "integer", "float64": "real", "object": "string", "str": "string"}
REMOTE_TYPE = {"string": 129, "integer": 20, "real": 5}
# numeric columns that are labels or flags, not amounts: declare them as dimensions so Tableau does not sum them
AS_DIMENSION = {"scored_customers": ["test_index", "churned_actual", "NumOfProducts", "IsActiveMember", "HasCrCard", "zero_balance",
                                     "Tenure", "target_unconstrained", "target_capacity_10pct", "target_capacity_15pct"],
                "drivers": ["rank"], "drift": ["simulated"]}


def _id(seed, n=28):
    return hashlib.sha1(seed.encode()).hexdigest()[:n]


def _uuid(seed):
    return "{" + str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest())).upper() + "}"


def sub(parent, tag, text=None, **attrs):
    """Add a child element; Python '_' in attribute names becomes Tableau's '-'."""
    el = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})
    if text is not None:
        el.text = text
    return el


def q(value):
    return f'"{value}"'


class Calc:
    def __init__(self, key, caption, formula, datatype, role, fmt=None):
        self.name = f"[Calculation_{int(_id(key, 14), 16) % 10**16:016d}]"
        self.caption, self.formula, self.datatype, self.role, self.fmt = caption, formula, datatype, role, fmt
        self.type = "quantitative" if role == "measure" else "nominal"

    def element(self, parent):
        attrs = dict(caption=self.caption, datatype=self.datatype, name=self.name, role=self.role, type=self.type)
        if self.fmt:
            attrs["default_format"] = self.fmt
        col = sub(parent, "column", **attrs)
        sub(col, "calculation", **{"class": "tableau", "formula": self.formula})
        return col


class Param:
    def __init__(self, caption, datatype, value, members=None, fmt=None, rng=None):
        self.caption, self.name, self.datatype, self.value, self.members, self.fmt, self.rng = caption, f"[{caption}]", datatype, value, members, fmt, rng

    @property
    def ref(self):
        return f"[{PARAM_DS}].{self.name}"

    def element(self, parent):
        literal = q(self.value) if self.datatype == "string" else str(self.value)
        attrs = dict(caption=self.caption, datatype=self.datatype, name=self.name, param_domain_type="list" if self.members else "range",
                     role="measure", type="nominal" if self.members else "quantitative", value=literal)
        if self.fmt:
            attrs["default_format"] = self.fmt
        col = sub(parent, "column", **attrs)
        sub(col, "calculation", **{"class": "tableau", "formula": literal})
        if self.members:
            members = sub(col, "members")
            for m in self.members:
                sub(members, "member", value=q(m))
        else:
            sub(col, "range", granularity=self.rng[2], max=self.rng[1], min=self.rng[0])
        return col


COST = Param("Cost scenario", "string", "base", members=["best", "base", "worst"])
CAPACITY = Param("Capacity", "string", "10%", members=["10%", "15%", "Unconstrained"])
PSI_MODERATE = Param("PSI moderate threshold", "real", 0.1, fmt="n0.00", rng=(0.0, 1.0, 0.01))
PSI_SIGNIFICANT = Param("PSI significant threshold", "real", 0.25, fmt="n0.00", rng=(0.0, 1.0, 0.01))
PARAMS = [COST, CAPACITY, PSI_MODERATE, PSI_SIGNIFICANT]


class Source:
    """One CSV as a Tableau data source, plus the calculated fields defined on it."""

    def __init__(self, stem, caption):
        self.stem, self.caption = stem, caption
        self.path = CSV_DIR / f"{stem}.csv"
        df = pd.read_csv(self.path)
        self.columns = [(c, DTYPE[str(t)]) for c, t in df.dtypes.items()]
        self.dims = set(AS_DIMENSION.get(stem, []))
        self.name = f"federated.{_id(stem)}"
        self.conn = f"textscan.{_id(stem + 'conn')}"
        self.calcs = {}

    def calc(self, key, caption, formula, datatype, role, fmt=None):
        self.calcs[key] = Calc(f"{self.stem}.{key}", caption, formula, datatype, role, fmt)
        return self.calcs[key]

    def field_type(self, column):
        dt = dict(self.columns)[column]
        if dt == "string" or column in self.dims:
            return dt, "dimension", "nominal" if dt == "string" else "ordinal"
        return dt, "measure", "quantitative"

    def column_decl(self, parent, column):
        dt, role, typ = self.field_type(column)
        sub(parent, "column", datatype=dt, name=f"[{column}]", role=role, type=typ)

    def element(self, parent):
        ds = sub(parent, "datasource", caption=self.caption, inline="true", name=self.name, version=VERSION)
        conn = sub(ds, "connection", **{"class": "federated"})
        named = sub(sub(conn, "named-connections"), "named-connection", caption=self.stem, name=self.conn)
        sub(named, "connection", **{"class": "textscan", "directory": REL_CSV_DIR, "filename": f"{self.stem}.csv", "password": "", "server": ""})
        rel = sub(conn, "relation", connection=self.conn, name=f"{self.stem}.csv", table=f"[{self.stem}#csv]", type="table")
        cols = sub(rel, "columns", character_set="UTF-8", header="yes", locale="en_US", separator=",")
        for i, (c, dt) in enumerate(self.columns):
            sub(cols, "column", datatype=dt, name=c, ordinal=i)
        meta = sub(conn, "metadata-records")
        for i, (c, dt) in enumerate(self.columns):
            rec = sub(meta, "metadata-record", **{"class": "column"})
            sub(rec, "remote-name", c)
            sub(rec, "remote-type", str(REMOTE_TYPE[dt]))
            sub(rec, "local-name", f"[{c}]")
            sub(rec, "parent-name", f"[{self.stem}.csv]")
            sub(rec, "remote-alias", c)
            sub(rec, "ordinal", str(i))
            sub(rec, "local-type", dt)
            sub(rec, "aggregation", "Count" if dt == "string" else "Sum")
            if dt == "string":
                sub(rec, "scale", "1")
                sub(rec, "width", "1073741823")
            sub(rec, "contains-null", "true")
            if dt == "string":
                sub(rec, "collation", flag="0", name="LEN_RUS")
        sub(ds, "aliases", enabled="yes")
        for c, _ in self.columns:
            if c in self.dims:
                self.column_decl(ds, c)
        for calc in self.calcs.values():
            calc.element(ds)
        return ds


class Sheet:
    """A worksheet under construction: tracks which fields and instances it uses so the XML dependencies stay complete."""

    def __init__(self, name, title, mark="Automatic"):
        self.name, self.title, self.mark = name, title, mark
        self.sources, self.deps, self.params = {}, {}, []   # deps: source -> {instance name: (column, derivation, type)}
        self.rows = self.cols = ""
        self.color = self.detail = self.text = None
        self.filters, self.palette, self.sort, self.reflines = [], None, [], []

    def use(self, source):
        self.sources[source.name] = source
        self.deps.setdefault(source.name, {})
        return source

    def inst(self, source, column, derivation, suffix, typ):
        """Register [derivation:column:suffix] on this sheet and return its fully qualified reference."""
        self.use(source)
        name = f"[{derivation_prefix(derivation)}:{column}:{suffix}]"
        self.deps[source.name][name] = (column, derivation, typ)
        return f"[{source.name}].{name}"

    def dim(self, source, column):                       # a string or flag column used as a discrete field
        _, _, typ = source.field_type(column)
        return self.inst(source, column, "None", "nk" if typ == "nominal" else "ok", typ)

    def total(self, source, column):                     # SUM of a raw measure
        return self.inst(source, column, "Sum", "qk", "quantitative")

    def calc_ref(self, source, calc):                    # a calculated field, as a measure or a discrete dimension
        if calc.role == "measure":
            return self.inst(source, calc.name[1:-1], "User", "qk", "quantitative")
        return self.inst(source, calc.name[1:-1], "User", "nk", "nominal")

    def use_param(self, *params):
        self.params += [p for p in params if p not in self.params]

    def element(self, parent):
        ws = sub(parent, "worksheet", name=self.name)
        title = sub(sub(ws, "layout-options"), "title")
        sub(sub(title, "formatted-text"), "run", self.title, bold="true", fontsize="12", fontcolor=INK)
        table = sub(ws, "table")
        view = sub(table, "view")
        dss = sub(view, "datasources")
        for s in self.sources.values():
            sub(dss, "datasource", caption=s.caption, name=s.name)
        if self.params:
            sub(dss, "datasource", name=PARAM_DS)
            dep = sub(view, "datasource-dependencies", datasource=PARAM_DS)
            for p in self.params:
                p.element(dep)
        for s in self.sources.values():
            dep = sub(view, "datasource-dependencies", datasource=s.name)
            used = {c for c, _, _ in self.deps[s.name].values()}
            for key, calc in s.calcs.items():
                if calc.name[1:-1] in used:
                    calc.element(dep)
            for c, _ in s.columns:
                if c in used:
                    s.column_decl(dep, c)
            for name, (column, derivation, typ) in self.deps[s.name].items():
                sub(dep, "column-instance", column=f"[{column}]", derivation=derivation, name=name, pivot="key", type=typ)
        for column, member in self.filters:
            f = sub(view, "filter", **{"class": "categorical", "column": column})
            sub(f, "groupfilter", function="member", level=column.split("].", 1)[1], member=q(member))
        for item in self.sort:
            view.append(item)
        if self.filters:
            slices = sub(view, "slices")
            for column, _ in self.filters:
                sub(slices, "column", column)
        sub(view, "aggregation", value="true")
        style = sub(table, "style")
        if self.palette:
            rule = sub(style, "style-rule", element="mark")
            field, mapping = self.palette
            enc = sub(rule, "encoding", attr="color", field=field.split("].", 1)[1], type="palette")
            for member, color in mapping.items():
                sub(sub(enc, "map", to=color), "bucket", q(member) if isinstance(member, str) else str(member))
        panes = sub(table, "panes")
        pane = sub(panes, "pane", selection_relaxation_option="selection-relaxation-allow")
        sub(sub(pane, "view"), "breakdown", value="auto")
        sub(pane, "mark", **{"class": self.mark})
        if self.color or self.detail or self.text:
            enc = sub(pane, "encodings")
            if self.color:
                sub(enc, "color", column=self.color)
            if self.detail:
                sub(enc, "lod", column=self.detail)
            if self.text:
                sub(enc, "text", column=self.text)
        for i, (axis, param, label) in enumerate(self.reflines):
            sub(pane, "reference-line", axis_column=axis, enable_instant_analytics="true", formula="average", id=f"refline{i}",
                label=label, label_type="custom", probability="95", scope="per-pane", value_column=param.ref, z_order=i + 1)
        sub(table, "rows", self.rows)
        sub(table, "cols", self.cols)
        sub(ws, "simple-id", uuid=_uuid(self.name))
        return ws


def derivation_prefix(derivation):
    return {"None": "none", "Sum": "sum", "User": "usr"}[derivation]


def computed_sort(column, using, direction="DESC"):
    return ET.Element("computed-sort", column=column, direction=direction, using=using)


def manual_sort(column, order):
    el = ET.Element("manual-sort", column=column, direction="ASC")
    d = sub(el, "dictionary")
    for m in order:
        sub(d, "bucket", q(m))
    return el


# ------------------------------------------------------------------------------------------------ the workbook
def build_sheets():
    seg, drv = Source("segment_summary", "segment_summary"), Source("drivers", "drivers")
    scored, scen, drift = Source("scored_customers", "scored_customers"), Source("scenarios", "scenarios"), Source("drift", "drift")
    sources = [seg, drv, scored, scen, drift]

    pick = f"[cost_scenario] = {COST.ref} AND [capacity] = {CAPACITY.ref} AND [basis] = {q('per_1000')}"
    geo = q("Geography")   # each dimension in segment_summary covers all customers once; Geography is used for the overall figures
    seg_churn = seg.calc("churn", "Churn rate", "SUM([churners]) / SUM([customers])", "real", "measure", "p0.0%")
    seg_kpi_churn = seg.calc("kpi_churn", "Overall churn rate", f"SUM(IF [dimension] = {geo} THEN [churners] END) / SUM(IF [dimension] = {geo} THEN [customers] END)",
                             "real", "measure", "p0.0%")
    seg_kpi_bar = seg.calc("kpi_bar", "Balance at risk share", f"SUM(IF [dimension] = {geo} THEN [balance_at_risk] END) / SUM(IF [dimension] = {geo} THEN [segment_balance] END)",
                           "real", "measure", "p0.0%")
    scen_net = scen.calc("net", "Net value per 1,000 (selected)", f"SUM(IF {pick} THEN [net_value] END)", "real", "measure", "n#,##0")
    scen_roi = scen.calc("roi", "ROI (selected)", f"SUM(IF {pick} THEN [net_value] END) / SUM(IF {pick} THEN [campaign_cost] END)", "real", "measure", "p0%")
    scen_net_cap = scen.calc("net_cap", "Net value per 1,000 at selected capacity",
                             f"SUM(IF [capacity] = {CAPACITY.ref} AND [basis] = {q('per_1000')} THEN [net_value] END)", "real", "measure", "n#,##0")
    scen_is_sel = scen.calc("is_sel", "Selected cost scenario", f"IF [cost_scenario] = {COST.ref} THEN {q('Selected')} ELSE {q('Other')} END", "string", "dimension")
    sc_sel = scored.calc("sel", "In selected cost scenario", f"IF [scenario] = {COST.ref} THEN {q('Selected')} ELSE {q('Other')} END", "string", "dimension")
    sc_flag = scored.calc("flag", "Campaign target",
                          f"IF (CASE {CAPACITY.ref} WHEN {q('10%')} THEN [target_capacity_10pct] WHEN {q('15%')} THEN [target_capacity_15pct] ELSE [target_unconstrained] END) = 1 "
                          f"THEN {q('Targeted')} ELSE {q('Not targeted')} END", "string", "dimension")

    def kpi(name, title, source, calc, params=()):
        s = Sheet(name, title)
        s.text = s.calc_ref(source, calc)
        s.use_param(*params)
        return s

    sheets = [
        kpi("KPI Churn rate", "Overall churn rate (all 10,000 customers)", seg, seg_kpi_churn),
        kpi("KPI Balance at risk", "Balance at risk: share of the bank's total balance held by churned customers", seg, seg_kpi_bar),
        kpi("KPI Net value", "Net value per 1,000 customers: selected scenario and capacity (units; assumptions)", scen, scen_net, (COST, CAPACITY)),
        kpi("KPI ROI", "ROI: selected scenario and capacity (assumptions)", scen, scen_roi, (COST, CAPACITY)),
    ]

    s = Sheet("Churn by segment", "Churn rate by segment (all 10,000 customers)", "Bar")
    d, g = s.dim(seg, "dimension"), s.dim(seg, "segment")
    s.rows, s.cols, s.color = f"({d} / {g})", s.calc_ref(seg, seg_churn), d
    sheets.append(s)

    s = Sheet("Churn drivers", "Churn drivers: mean |SHAP| by feature, coloured by lever (test set)", "Bar")
    feat, lever, shap = s.dim(drv, "feature"), s.dim(drv, "lever"), s.total(drv, "mean_abs_shap")
    s.rows, s.cols, s.color = feat, shap, lever
    s.sort = [computed_sort(feat, shap)]
    s.palette = (lever, {"Customer": "#2a78d6", "Product": "#eb6834", "Engagement": "#1baf7a"})
    sheets.append(s)

    s = Sheet("Scenario comparison", "Net value per 1,000 customers by cost scenario, at the selected capacity (units; assumptions)", "Bar")
    s.use_param(CAPACITY, COST)
    scn = s.dim(scen, "cost_scenario")
    s.cols, s.rows, s.color = scn, s.calc_ref(scen, scen_net_cap), s.calc_ref(scen, scen_is_sel)
    s.sort = [manual_sort(scn, ["best", "base", "worst"])]
    s.palette = (s.color, {"Selected": BLUE, "Other": GREY})
    sheets.append(s)

    s = Sheet("Risk vs value", "Risk vs customer value: test customers in the selected cost scenario, coloured by campaign target at the selected capacity", "Circle")
    s.use_param(COST, CAPACITY)
    sel = s.calc_ref(scored, sc_sel)
    s.cols, s.rows = s.total(scored, "risk"), s.total(scored, "customer_value")
    s.detail, s.color = s.dim(scored, "test_index"), s.calc_ref(scored, sc_flag)
    s.filters = [(sel, "Selected")]
    s.palette = (s.color, {"Targeted": BLUE, "Not targeted": GREY})
    sheets.append(s)

    s = Sheet("Drift", "Drift: PSI by model input and score. Actual (train vs test) and SIMULATED scenarios (the dataset has no time column)", "Bar")
    s.use_param(PSI_MODERATE, PSI_SIGNIFICANT)
    name, desc, band, psi = s.dim(drift, "name"), s.dim(drift, "description"), s.dim(drift, "band"), s.total(drift, "psi")
    s.cols, s.rows, s.color = f"({desc} * {psi})", name, band
    s.sort = [computed_sort(name, psi)]
    s.palette = (band, {"stable": GREEN, "moderate": AMBER, "significant": RED})
    s.reflines = [(psi, PSI_MODERATE, "Moderate <Value>"), (psi, PSI_SIGNIFICANT, "Significant <Value>")]
    sheets.append(s)
    return sources, sheets


NOTE = ("Money is in units and rests on assumptions (src/config/retention_assumptions.yaml), not bank figures. Drift scenarios are SIMULATED: "
        "the dataset has no time column, so only the train-vs-test column is a real comparison.")


def dashboard_xml(parent, sources, sheets):
    names = {s.name: s for s in sheets}
    dash = sub(parent, "dashboard", name=DASHBOARD)
    sub(sub(sub(dash, "layout-options"), "title"), "formatted-text")
    sub(dash.find("layout-options/title/formatted-text"), "run", DASHBOARD, fontsize="16", bold="true")
    sub(dash, "size", maxheight="1500", minheight="1500", sizing_mode="vscroll")
    dss = sub(dash, "datasources")
    for src in sources:
        sub(dss, "datasource", caption=src.caption, name=src.name)
    sub(dss, "datasource", name=PARAM_DS)
    dep = sub(dash, "datasource-dependencies", datasource=PARAM_DS)
    for p in (COST, CAPACITY):
        p.element(dep)

    ids = iter(range(3, 1000))
    zones = sub(dash, "zones")
    root = sub(zones, "zone", h=100000, id=1, type_v2="layout-basic", w=100000, x=0, y=0)
    flow = sub(root, "zone", h=100000, id=2, param="vert", type_v2="layout-flow", w=100000, x=0, y=0)
    rows = [("title", 4000), ("params", 7000), ("kpi", 12000), ("mid1", 25000), ("mid2", 25000), ("drift", 27000)]
    y = 0
    for kind, h in rows:
        row = sub(flow, "zone", h=h, id=next(ids), param="horz", type_v2="layout-flow", w=100000, x=0, y=y)

        def put(width, x, **attrs):
            return sub(row, "zone", h=h, id=next(ids), w=width, x=x, y=y, **attrs)

        if kind == "title":
            put(100000, 0, type_v2="title")
        elif kind == "params":
            put(20000, 0, mode="compact", param=COST.ref, type_v2="paramctrl")
            put(20000, 20000, mode="compact", param=CAPACITY.ref, type_v2="paramctrl")
            text = put(60000, 40000, type_v2="text")
            sub(sub(text, "formatted-text"), "run", NOTE, fontsize="9", fontcolor="#52514e")
        elif kind == "kpi":
            for i, nm in enumerate(["KPI Churn rate", "KPI Balance at risk", "KPI Net value", "KPI ROI"]):
                put(25000, 25000 * i, name=nm)
        elif kind == "mid1":
            put(50000, 0, name="Churn by segment")
            put(50000, 50000, name="Churn drivers")
        elif kind == "mid2":
            put(35000, 0, name="Scenario comparison")
            put(65000, 35000, name="Risk vs value")
        else:
            put(100000, 0, name="Drift")
        y += h
    missing = {z.get("name") for z in zones.iter("zone") if z.get("name")} - set(names)
    assert not missing, missing
    sub(dash, "simple-id", uuid=_uuid(DASHBOARD))


def windows_xml(parent, sheets):
    win = sub(parent, "windows", source_height="30")
    for s in sheets:
        w = sub(win, "window", **{"class": "worksheet", "name": s.name})
        cards = sub(w, "cards")
        left = sub(sub(cards, "edge", name="left"), "strip", size="160")
        for t in ("pages", "filters", "marks"):
            sub(left, "card", type=t)
        top = sub(cards, "edge", name="top")
        sub(sub(top, "strip", size="2147483647"), "card", type="columns")
        sub(sub(top, "strip", size="2147483647"), "card", type="rows")
        sub(sub(top, "strip", size="31"), "card", type="title")
        if s.color and s.palette:
            sub(sub(sub(cards, "edge", name="right"), "strip", size="160"), "card", pane_specification_id="0", param=s.color, type="color")
        sub(w, "simple-id", uuid=_uuid("window" + s.name))
    dash = sub(win, "window", **{"class": "dashboard", "maximized": "true", "name": DASHBOARD})
    views = sub(dash, "viewpoints")
    for s in sheets:
        sub(sub(views, "viewpoint", name=s.name), "zoom", type="entire-view")
    sub(dash, "active", id="-1")
    sub(dash, "simple-id", uuid=_uuid("window" + DASHBOARD))


def build():
    sources, sheets = build_sheets()
    wb = ET.Element("workbook", {"original-version": VERSION, "source-build": SOURCE_BUILD, "source-platform": "win", "version": VERSION,
                                 "xmlns:user": "http://www.tableausoftware.com/xml/user"})
    sub(wb, "document-format-change-manifest")
    sub(wb, "preferences")
    dss = sub(wb, "datasources")
    pds = sub(dss, "datasource", hasconnection="false", inline="true", name=PARAM_DS, version=VERSION)
    sub(pds, "aliases", enabled="yes")
    for p in PARAMS:
        p.element(pds)
    for src in sources:
        src.element(dss)
    wss = sub(wb, "worksheets")
    for s in sheets:
        s.element(wss)
    dashboard_xml(sub(wb, "dashboards"), sources, sheets)
    windows_xml(wb, sheets)
    explain = sub(wb, "explain-data", enabled_for_viewer="false", extreme_values_enabled_for_all="false")   # Explain Data defaults, as in a saved workbook
    kinds = sub(explain, "explanation-types")
    for kind in ("number-of-records", "average-of-records", "aggregated-dimensions", "distribution-of-records", "unvisualized-measures",
                 "extreme-values", "null-value", "tvd-single-value", "ad-single-value"):
        sub(kinds, "explanation-type", enabled="true", type=kind)
    ET.indent(wb, space="  ")
    return wb


def write(path=OUT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(build()).write(path, encoding="utf-8", xml_declaration=True)
    return path


# ------------------------------------------------------------------------------------------------ validation
def validate(path=OUT_PATH, csv_dir=CSV_DIR):
    """Return a list of problems (empty = clean): well-formed, every CSV field exists, every reference inside the workbook resolves."""
    problems = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]
    csv_dir = Path(csv_dir)
    param_names = {c.get("name") for c in root.find("datasources/datasource[@name='Parameters']").findall("column")}
    ds_cols, ds_calcs, ds_names = {}, {}, {}
    for ds in root.find("datasources"):
        if ds.get("name") == PARAM_DS:
            continue
        conn = ds.find("connection/named-connections/named-connection/connection")
        file = csv_dir / conn.get("filename")
        if not file.exists():
            problems.append(f"{ds.get('caption')}: {file.name} not found")
            continue
        header = list(pd.read_csv(file, nrows=1).columns)
        declared = [c.get("name") for c in ds.findall("connection/relation/columns/column")]
        if declared != header:
            problems.append(f"{file.name}: workbook columns {declared} differ from CSV header {header}")
        meta = [r.findtext("remote-name") for r in ds.findall("connection/metadata-records/metadata-record")]
        if meta != header:
            problems.append(f"{file.name}: metadata records differ from CSV header")
        ds_cols[ds.get("name")], ds_names[ds.get("name")] = set(header), file.name
        ds_calcs[ds.get("name")] = {c.get("name"): c.find("calculation").get("formula") for c in ds.findall("column") if c.find("calculation") is not None}

    def check_formula(owner, formula, cols, calcs):
        for p in re.findall(r"\[Parameters\]\.(\[[^\]]+\])", formula):
            if p not in param_names:
                problems.append(f"{owner}: unknown parameter {p}")
        rest = re.sub(r"\[Parameters\]\.\[[^\]]+\]", "", formula)
        for token in re.findall(r"\[([^\]]+)\]", rest):
            if token not in cols and f"[{token}]" not in calcs:
                problems.append(f"{owner}: formula uses [{token}], which is not a CSV column or calculated field")

    for name, calcs in ds_calcs.items():
        for cname, formula in calcs.items():
            check_formula(f"{ds_names[name]} {cname}", formula, ds_cols[name], calcs)

    sheets = {}
    for ws in root.find("worksheets"):
        sheets[ws.get("name")] = ws
        table = ws.find("table")
        view = table.find("view")
        instances = {}
        for dep in view.findall("datasource-dependencies"):
            dsn = dep.get("datasource")
            declared = {c.get("name") for c in dep.findall("column")}
            instances[dsn] = {i.get("name") for i in dep.findall("column-instance")}
            for c in dep.findall("column"):
                if dsn != PARAM_DS and c.find("calculation") is None and c.get("name")[1:-1] not in ds_cols.get(dsn, set()):
                    problems.append(f"sheet '{ws.get('name')}': column {c.get('name')} is not in {ds_names.get(dsn, dsn)}")
            for i in dep.findall("column-instance"):
                if i.get("column") not in declared:
                    problems.append(f"sheet '{ws.get('name')}': instance {i.get('name')} points at undeclared column {i.get('column')}")
        sheet_params = {c.get("name") for dep in view.findall("datasource-dependencies") if dep.get("datasource") == PARAM_DS for c in dep.findall("column")}
        for dep in view.findall("datasource-dependencies"):
            for c in dep.findall("column"):
                formula = c.find("calculation").get("formula") if c.find("calculation") is not None else ""
                for p in re.findall(r"\[Parameters\]\.(\[[^\]]+\])", formula):
                    if p not in sheet_params:
                        problems.append(f"sheet '{ws.get('name')}': calculated field {c.get('caption')} uses parameter {p}, which the sheet does not declare")
        pane = table.find("panes/pane")
        refs = [t for tag in ("rows", "cols") for t in re.findall(r"\[[^\]]+\]\.\[[^\]]+\]", table.findtext(tag) or "")]
        encodings = pane.find("encodings")
        refs += [e.get("column") for e in (encodings if encodings is not None else [])]
        refs += [e.get("column") for e in view.findall("filter")] + [e.get("column") for e in view.findall("computed-sort")]
        refs += [e.get("column") for e in view.findall("manual-sort")]
        refs += [e.get("axis-column") for e in pane.findall("reference-line")]
        for ref in refs:
            dsn, inst = re.match(r"\[([^\]]+)\]\.(\[.+\])$", ref).groups()
            if inst not in instances.get(dsn, set()):
                problems.append(f"sheet '{ws.get('name')}': {ref} is not defined in the sheet's data source dependencies")
        for e in pane.findall("reference-line"):
            if e.get("value-column").split(".", 1)[1] not in param_names or not e.get("value-column").startswith(f"[{PARAM_DS}]."):
                problems.append(f"sheet '{ws.get('name')}': reference line value {e.get('value-column')} is not a parameter")

    for dash in root.find("dashboards"):
        for z in dash.iter("zone"):
            if z.get("name") and z.get("name") not in sheets:
                problems.append(f"dashboard zone '{z.get('name')}' is not a worksheet")
            if z.get("type-v2") == "paramctrl" and z.get("param").split(".", 1)[1] not in param_names:
                problems.append(f"dashboard parameter control {z.get('param')} is not a parameter")
    for w in root.find("windows"):
        if w.get("name") not in sheets and w.get("name") != DASHBOARD:
            problems.append(f"window '{w.get('name')}' has no matching sheet")
    return problems


if __name__ == "__main__":
    out = write()
    issues = validate(out)
    print(f"wrote {out}")
    print("validation: " + ("clean" if not issues else f"{len(issues)} problem(s)"))
    for issue in issues:
        print("  -", issue)
