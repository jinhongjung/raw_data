import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import html
import numpy as np
import ipywidgets as widgets
import plotly.io as pio

from IPython.display import clear_output

from google.colab import output
output.enable_custom_widget_manager()

# Plotly renderer for Google Colab
pio.renderers.default = "colab"



def plot_similarity_analysis(verification_df):
    # --------------------------------------------------
    # Prepare error statistics
    # --------------------------------------------------
    mae = verification_df["error"].mean()
    median_error = verification_df["error"].median()
    std_error = verification_df["error"].std()
    max_error = verification_df["error"].max()


    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Exact vs. MinHash Similarity",
            "Approximation Error Distribution"
        ),
        horizontal_spacing=0.12
    )


    # --------------------------------------------------
    # 1. Interactive Scatter Plot
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=verification_df["exact_similarity"],
            y=verification_df["approx_similarity"],
            mode="markers",

            # Information shown when hovering over a point
            customdata=verification_df[
                ["doc_a", "doc_b", "error"]
            ].to_numpy(),

            hovertemplate=(
                "<b>Pair: (%{customdata[0]}, %{customdata[1]})</b><br>"
                "Exact Jaccard: %{x:.3f}<br>"
                "MinHash Approx.: %{y:.3f}<br>"
                "Absolute Error: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),

            marker=dict(
                size=8,
                opacity=0.7
            ),

            name="Candidate Pair"
        ),
        row=1,
        col=1
    )

    # y = x reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            hoverinfo="skip",
            name="y = x"
        ),
        row=1,
        col=1
    )


    # --------------------------------------------------
    # 2. Error Histogram
    # --------------------------------------------------
    fig.add_trace(
        go.Histogram(
            x=verification_df["error"],
            nbinsx=20,
            hovertemplate=(
                "Error range: %{x}<br>"
                "Count: %{y}"
                "<extra></extra>"
            ),
            marker=dict(
                line=dict(
                    color="black",
                    width=1
                )
            ),
            name="Absolute Error"
        ),
        row=1,
        col=2
    )


    # --------------------------------------------------
    # Error Statistics
    # --------------------------------------------------
    stats_text = (
        f"<b>Error Statistics</b><br>"
        f"MAE: {mae:.3f}<br>"
        f"Median: {median_error:.3f}<br>"
        f"Std: {std_error:.3f}<br>"
        f"Max: {max_error:.3f}"
    )

    fig.add_annotation(
        x=0.97,
        y=0.97,
        xref="x2 domain",
        yref="y2 domain",
        text=stats_text,
        showarrow=False,
        align="left",
        xanchor="right",
        yanchor="top",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255,255,255,0.8)"
    )


    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_xaxes(
        title_text="Exact Jaccard Similarity",
        range=[0, 1],
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Approximate Similarity",
        range=[0, 1],
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="Absolute Error",
        row=1,
        col=2
    )

    fig.update_yaxes(
        title_text="Number of Candidate Pairs",
        row=1,
        col=2
    )

    fig.update_layout(
        width=1200,
        height=500,
        template="plotly_white",
        showlegend=False,
        hovermode="closest"
    )

    fig.show()


def display_candidates(candidates_df, info_df):
    cards = []

    for rank, (_, row) in enumerate(candidates_df.iterrows(), 1):
        doc_a = int(row["doc_a"])
        doc_b = int(row["doc_b"])

        exact_sim = row["exact_similarity"]
        approx_sim = row["approx_similarity"]
        error = abs(exact_sim - approx_sim)

        document_cards = []

        for doc_id in [doc_a, doc_b]:
            info = info_df.loc[doc_id]

            title = str(info["title"])
            category = str(info["category"])
            description = str(info["description"]).replace("\n", " ")[:400]

            document_cards.append(f"""
                <div style="
                    flex:1;
                    border:1px solid #ddd;
                    border-radius:10px;
                    padding:16px;
                    min-width:0;
                ">
                    <div style="font-size:13px; color:#666;">
                        Document {doc_id}
                    </div>

                    <div style="
                        font-size:17px;
                        font-weight:600;
                        margin:6px 0 10px 0;
                    ">
                        {html.escape(title)}
                    </div>

                    <div style="
                        display:inline-block;
                        padding:4px 8px;
                        border-radius:6px;
                        background:#f2f2f2;
                        font-size:12px;
                        margin-bottom:12px;
                    ">
                        {html.escape(category)}
                    </div>

                    <div style="
                        font-size:13px;
                        line-height:1.5;
                        color:#444;
                    ">
                        {html.escape(description)}...
                    </div>
                </div>
            """)

        cards.append(f"""
            <div style="
                border:1px solid #ccc;
                border-radius:12px;
                padding:18px;
                margin-bottom:18px;
            ">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:14px;
                ">
                    <div style="font-size:18px; font-weight:700;">
                        #{rank} Pair ({doc_a}, {doc_b})
                    </div>

                    <div style="font-size:14px;">
                        <b>Exact</b>: {exact_sim:.3f}
                        &nbsp;&nbsp;
                        <b>Approx</b>: {approx_sim:.3f}
                        &nbsp;&nbsp;
                        <b>Error</b>: {error:.3f}
                    </div>
                </div>

                <div style="display:flex; gap:14px;">
                    {''.join(document_cards)}
                </div>
            </div>
        """)

    display(HTML("".join(cards)))


def interactive_lsh_probability():

    # Sliders
    b_slider = widgets.IntSlider(
        value=20,
        min=1,
        max=50,
        step=1,
        description="Bands (b):",
        continuous_update=False
    )

    r_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=10,
        step=1,
        description="Rows (r):",
        continuous_update=False
    )

    # Output area for the plot
    plot_output = widgets.Output()

    def update_plot(change=None):

        b = b_slider.value
        r = r_slider.value

        # Jaccard similarity: 0 ~ 1
        s = np.linspace(0, 1, 501)

        # Candidate probability
        probability = 1 - (1 - s**r)**b

        # Approximate threshold
        threshold = (1 / b) ** (1 / r)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=s,
                y=probability,
                mode="lines",
                hovertemplate=(
                    "Jaccard Similarity: %{x:.3f}<br>"
                    "Candidate Probability: %{y:.3f}"
                    "<extra></extra>"
                )
            )
        )

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            annotation_text=f"threshold ≈ {threshold:.2f}",
            annotation_position="top"
        )

        fig.update_layout(
            title=(
                f"LSH Candidate Probability "
                f"(b={b}, r={r}, hashes={b*r})"
            ),
            xaxis_title="Jaccard Similarity (s)",
            yaxis_title="P(candidate)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=850,
            height=500,
            template="plotly_white",
            showlegend=False
        )

        # Redraw the figure
        with plot_output:
            clear_output(wait=True)
            fig.show()

    # Update when sliders change
    b_slider.observe(update_plot, names="value")
    r_slider.observe(update_plot, names="value")

    # Display UI
    display(
        widgets.HBox([b_slider, r_slider]),
        plot_output
    )

    # Initial plot
    update_plot()


def plot_similarity_distributions(sensitivity_similarity_df):
    fig = px.box(
        sensitivity_similarity_df,
        x="exact_similarity",
        y="setting",
        color="setting",
        points="all",
        title="Exact Similarity of LSH Candidates",
    )

    fig.update_traces(
        jitter=0.25,
        pointpos=0,
        marker=dict(size=5, opacity=0.45),
    )

    fig.update_layout(
        xaxis_title="Exact Jaccard Similarity",
        yaxis_title="LSH Setting",
        xaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=1000,
        height=450,
        showlegend=False,
    )

    fig.show()
