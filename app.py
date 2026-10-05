import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

CATS = ["gender", "race/ethnicity", "parental level of education",
        "lunch", "test preparation course"]
SUBJECTS = {"math score": "Mathematics",
            "reading score": "Reading",
            "writing score": "Writing"}
PASS_MARK = 40

PAGES = ["🏠 Home",
         "🔮 Predict My Result",
         "🔎 Student Lookup",
         "🎛️ What-If Simulator",
         "📊 Data Explorer",
         "📈 Class Insights"]


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    bundle = joblib.load("student_model.pkl")
    return bundle["model"], bundle["columns"]


@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df["average"] = df[list(SUBJECTS)].mean(axis=1)
    df["result"] = (df["average"] >= PASS_MARK).map({True: "Pass", False: "Fail"})
    df["weakest"] = df[list(SUBJECTS)].idxmin(axis=1).map(SUBJECTS)
    return df


model, columns = load_model()
df = load_data()


def options(col):
    """Dropdown options for a column. 'Associate's degree' is left out."""
    vals = sorted(df[col].unique())
    return [v for v in vals if "associate" not in v.lower()]


# ---------------------------------------------------------------- helpers
def predict(raw):
    """Return (pass_percentage, label) for one student."""
    X = pd.get_dummies(pd.DataFrame([raw]), columns=CATS)
    X = X.reindex(columns=columns, fill_value=0).astype(float)
    pass_pct = model.predict_proba(X)[0][1] * 100
    label = "Pass" if pass_pct >= 50 else "Fail"
    return pass_pct, label


def score_chart(scores, weakest, class_avg=None):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = list(scores.keys())
    x = np.arange(len(names))
    if class_avg:
        w = 0.38
        ax.bar(x - w / 2, [scores[n] for n in names], w,
               color=["#e45756" if n == weakest else "#4c78a8" for n in names],
               label="Student")
        ax.bar(x + w / 2, [class_avg[n] for n in names], w,
               color="#bab0ac", label="Class average")
        ax.legend()
    else:
        ax.bar(x, [scores[n] for n in names],
               color=["#e45756" if n == weakest else "#4c78a8" for n in names])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("Subject-wise scores")
    fig.tight_layout()
    return fig


def show_report(raw, compare=False):
    pass_pct, label = predict(raw)
    scores = {SUBJECTS[k]: raw[k] for k in SUBJECTS}
    weakest = min(scores, key=scores.get)

    st.subheader("📄 Personalized Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Result", label)
    c2.metric("Chance of Passing", f"{pass_pct:.1f}%")
    c3.metric("Weakest Subject", weakest)

    st.progress(int(pass_pct))
    if label == "Pass":
        st.success(f"Likely to pass. To improve further, focus more on {weakest}.")
    else:
        st.error(f"At risk of failing. Focus more on {weakest}.")

    class_avg = None
    if compare:
        class_avg = {SUBJECTS[k]: df[k].mean() for k in SUBJECTS}
    st.pyplot(score_chart(scores, weakest, class_avg))
    return pass_pct, label, weakest


def go(target):
    """Callback used by the Home page buttons to switch pages."""
    st.session_state["page"] = target


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎓 Student Performance")
page = st.sidebar.radio("Navigate", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.caption(f"Pass mark: average ≥ {PASS_MARK}  \nDataset: {len(df)} students")


# ================================================================ 1. HOME
if page == PAGES[0]:
    st.title("🎓 Student Performance Prediction & Recommendation")
    st.write("Predict whether a student will pass, find the weakest subject, "
             "and explore how the whole class is doing.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", len(df))
    k2.metric("Pass rate", f"{(df['result'] == 'Pass').mean() * 100:.1f}%")
    k3.metric("Class average", f"{df['average'].mean():.1f}")
    k4.metric("Most common weak subject", df["weakest"].mode()[0])

    st.markdown("### What can you do here?")
    cards = [
        (PAGES[1], "Enter your details and scores to get a personalized report."),
        (PAGES[2], "Pick a student ID and compare with the class average."),
        (PAGES[3], "Move sliders and watch the pass chance change live."),
        (PAGES[4], "Filter the dataset, plot it and download it."),
        (PAGES[5], "Pass/fail split, group comparisons and correlations."),
    ]
    cols = st.columns(3)
    for i, (target, desc) in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{target}**")
                st.caption(desc)
                st.button("Open →", key=f"open_{i}", on_click=go, args=(target,))


# ================================================================ 2. PREDICT
elif page == PAGES[1]:
    st.title("🔮 Predict My Result")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", options("gender"))
        race = st.selectbox("Race/Ethnicity", options("race/ethnicity"))
        parent = st.selectbox("Parental level of education",
                              options("parental level of education"))
        lunch = st.selectbox("Lunch type", options("lunch"))
        prep = st.selectbox("Test preparation course",
                            options("test preparation course"))
    with col2:
        math = st.slider("Math score", 0, 100, 60)
        reading = st.slider("Reading score", 0, 100, 60)
        writing = st.slider("Writing score", 0, 100, 60)

    if st.button("Generate my report", type="primary"):
        raw = {"gender": gender, "race/ethnicity": race,
               "parental level of education": parent, "lunch": lunch,
               "test preparation course": prep,
               "math score": math, "reading score": reading,
               "writing score": writing}
        pass_pct, label, weakest = show_report(raw, compare=True)

        report = pd.DataFrame([{**raw, "chance_of_passing_%": round(pass_pct, 1),
                                "predicted_result": label,
                                "weakest_subject": weakest}])
        st.download_button("⬇️ Download report (CSV)",
                           report.to_csv(index=False).encode("utf-8"),
                           "my_report.csv", "text/csv")


# ================================================================ 3. LOOKUP
elif page == PAGES[2]:
    st.title("🔎 Student Lookup")
    sid = st.number_input(f"Student ID (1 to {len(df)})", min_value=1,
                          max_value=len(df), value=1, step=1)
    row = df.iloc[int(sid) - 1]

    with st.expander("Student profile", expanded=True):
        p1, p2 = st.columns(2)
        for i, c in enumerate(CATS):
            (p1 if i % 2 == 0 else p2).write(f"**{c.title()}:** {row[c]}")

    show_report({c: row[c] for c in CATS + list(SUBJECTS)}, compare=True)

    pct_rank = (df["average"] < row["average"]).mean() * 100
    st.metric("Class percentile (by average score)", f"{pct_rank:.0f}th",
              help="Percentage of students with a lower average than this student.")
    st.caption(f"Actual average: {row['average']:.1f} → actual result: {row['result']}")


# ================================================================ 4. WHAT-IF
elif page == PAGES[3]:
    st.title("🎛️ What-If Simulator")
    st.write("Change anything below. The prediction updates instantly.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Background**")
        gender = st.selectbox("Gender", options("gender"), key="w_g")
        race = st.selectbox("Race/Ethnicity", options("race/ethnicity"), key="w_r")
        parent = st.selectbox("Parental education",
                              options("parental level of education"), key="w_p")
        lunch = st.selectbox("Lunch type", options("lunch"), key="w_l")
        prep = st.selectbox("Test preparation course",
                            options("test preparation course"), key="w_t")
    with right:
        st.markdown("**Scores**")
        math = st.slider("Math", 0, 100, 60, key="w_m")
        reading = st.slider("Reading", 0, 100, 60, key="w_rd")
        writing = st.slider("Writing", 0, 100, 60, key="w_w")

    raw = {"gender": gender, "race/ethnicity": race,
           "parental level of education": parent, "lunch": lunch,
           "test preparation course": prep,
           "math score": math, "reading score": reading, "writing score": writing}

    pass_pct, label = predict(raw)
    m1, m2 = st.columns(2)
    m1.metric("Predicted Result", label)
    m2.metric("Chance of Passing", f"{pass_pct:.1f}%")
    st.progress(int(pass_pct))

    st.markdown("#### How does the math score change the chance of passing?")
    xs = list(range(0, 101, 5))
    ys = [predict({**raw, "math score": v})[0] for v in xs]
    curve = pd.DataFrame({"Math score": xs, "Chance of passing (%)": ys}).set_index("Math score")
    st.line_chart(curve)


# ================================================================ 5. EXPLORER
elif page == PAGES[4]:
    st.title("📊 Data Explorer")

    f1, f2, f3, f4 = st.columns(4)
    g = f1.multiselect("Gender", sorted(df["gender"].unique()),
                       default=sorted(df["gender"].unique()))
    r = f2.multiselect("Race/Ethnicity", sorted(df["race/ethnicity"].unique()),
                       default=sorted(df["race/ethnicity"].unique()))
    l = f3.multiselect("Lunch", sorted(df["lunch"].unique()),
                       default=sorted(df["lunch"].unique()))
    t = f4.multiselect("Test prep", sorted(df["test preparation course"].unique()),
                       default=sorted(df["test preparation course"].unique()))

    s1, s2 = st.columns(2)
    avg_range = s1.slider("Average score range", 0, 100, (0, 100))
    res = s2.radio("Result", ["All", "Pass", "Fail"], horizontal=True)

    view = df[df["gender"].isin(g) & df["race/ethnicity"].isin(r)
              & df["lunch"].isin(l) & df["test preparation course"].isin(t)
              & df["average"].between(*avg_range)]
    if res != "All":
        view = view[view["result"] == res]

    if view.empty:
        st.warning("No students match these filters.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Students shown", len(view))
        m2.metric("Pass rate", f"{(view['result'] == 'Pass').mean() * 100:.1f}%")
        m3.metric("Average score", f"{view['average'].mean():.1f}")

        t_table, t_hist, t_scatter = st.tabs(["Table", "Distribution", "Scatter"])

        with t_table:
            st.dataframe(view, width="stretch")
            st.download_button("⬇️ Download filtered data (CSV)",
                               view.to_csv(index=False).encode("utf-8"),
                               "filtered_students.csv", "text/csv")

        with t_hist:
            subj = st.selectbox("Subject", list(SUBJECTS), format_func=SUBJECTS.get)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(view[subj], bins=15, color="#4c78a8", edgecolor="white")
            ax.set_xlabel(SUBJECTS[subj] + " score")
            ax.set_ylabel("Students")
            st.pyplot(fig)

        with t_scatter:
            x_ax = st.selectbox("X axis", list(SUBJECTS), index=0, format_func=SUBJECTS.get)
            y_ax = st.selectbox("Y axis", list(SUBJECTS), index=1, format_func=SUBJECTS.get)
            fig, ax = plt.subplots(figsize=(6, 4))
            for name, colour in [("Pass", "#59a14f"), ("Fail", "#e15759")]:
                part = view[view["result"] == name]
                ax.scatter(part[x_ax], part[y_ax], s=18, alpha=0.7,
                           color=colour, label=name)
            ax.set_xlabel(SUBJECTS[x_ax])
            ax.set_ylabel(SUBJECTS[y_ax])
            ax.legend()
            st.pyplot(fig)


# ================================================================ 6. INSIGHTS
elif page == PAGES[5]:
    st.title("📈 Class Insights")

    c1, c2 = st.columns(2)
    with c1:
        counts = df["result"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#59a14f" if i == "Pass" else "#e15759" for i in counts.index])
        ax1.set_title("Pass / Fail distribution")
        st.pyplot(fig1)
    with c2:
        weak = df["weakest"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(weak.index, weak.values, color="#4c78a8")
        ax2.set_ylabel("Number of students")
        ax2.set_title("How often each subject is the weakest")
        st.pyplot(fig2)

    st.markdown("### Compare groups")
    cat = st.selectbox("Group students by", CATS, format_func=str.title)
    pass_rate = (df.groupby(cat)["result"]
                 .apply(lambda s: (s == "Pass").mean() * 100)
                 .sort_values())
    avg_scores = df.groupby(cat)[list(SUBJECTS)].mean().rename(columns=SUBJECTS)

    g1, g2 = st.columns(2)
    with g1:
        st.write("**Pass rate (%)**")
        st.bar_chart(pass_rate)
    with g2:
        st.write("**Average subject scores**")
        st.bar_chart(avg_scores)

    st.markdown("### Do subjects move together?")
    corr = df[list(SUBJECTS)].corr()
    fig3, ax3 = plt.subplots(figsize=(4, 3.5))
    im = ax3.imshow(corr, cmap="Blues", vmin=0, vmax=1)
    labels = list(SUBJECTS.values())
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(labels)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig3.colorbar(im)
    st.pyplot(fig3)

    st.markdown("### Model Evaluation")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", width=450)
    else:
        st.info("confusion_matrix.png not found in the project folder.")
