# File: ui/visualization.py
# Location: ./ui/visualization.py

import streamlit as st
import numpy as np

# Check if visualization packages are available
try:
    import networkx as nx
    import plotly.graph_objects as go
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

def visualize_plan_execute_structure():
    """Plan-Execute 구조를 시각화"""
    if not VISUALIZATION_AVAILABLE:
        st.warning("📊 Visualization packages not installed. System structure:")
        
        # Text-based visualization
        st.markdown("""
        ```
        START → Planner → Router → Executor(s) → Re-planner → Router
                            ↑___________________________________|
                                                                ↓
                                                        Final Responder → END
        ```
        """)
        
        if st.session_state.executors:
            st.info(f"🤖 **Executors** ({len(st.session_state.executors)}): {', '.join(list(st.session_state.executors.keys())[:3])}...")
        else:
            st.warning("⚠️ **Executors**: None - Please create executors")
        return None
    
    # Create figure when visualization is available
    fig = go.Figure()
    
    # Node positions
    positions = {
        "START": (0, 0.5),
        "Planner": (1, 0.5),
        "Router": (2, 0.5),
        "Executors": (3, 0.5),
        "Re-planner": (4, 0.5),
        "Final Responder": (5, 0.3),
        "END": (6, 0.3)
    }
    
    # Add START node
    fig.add_trace(go.Scatter(
        x=[positions["START"][0]], y=[positions["START"][1]],
        mode='markers+text',
        marker=dict(symbol='circle', size=60, color="#4CAF50"),
        text=["START"],
        textposition="middle center",
        name="START"
    ))
    
    # Add system nodes
    system_nodes = ["Planner", "Router", "Re-planner", "Final Responder"]
    for node in system_nodes:
        fig.add_trace(go.Scatter(
            x=[positions[node][0]], y=[positions[node][1]],
            mode='markers+text',
            marker=dict(
                symbol='square',
                size=80,
                color="#2196F3" if node != "Final Responder" else "#9C27B0"
            ),
            text=[node],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            name=node
        ))
    
    # Add Executors
    if st.session_state.executors:
        executor_names = list(st.session_state.executors.keys())
        num_executors = len(executor_names)
        
        # Spread executors vertically
        y_positions = np.linspace(0.2, 0.8, num_executors)
        
        for i, executor in enumerate(executor_names):
            fig.add_trace(go.Scatter(
                x=[positions["Executors"][0]], y=[y_positions[i]],
                mode='markers+text',
                marker=dict(symbol='hexagon', size=60, color="#FF9800"),
                text=[executor[:10] + "..." if len(executor) > 10 else executor],
                textposition="middle center",
                textfont=dict(size=9),
                name=executor,
                hovertemplate=f"<b>{executor}</b><extra></extra>"
            ))
    else:
        # No executors placeholder
        fig.add_annotation(
            x=positions["Executors"][0], y=positions["Executors"][1],
            text="⚠️ No Executors",
            showarrow=False,
            font=dict(size=12, color="#FF5722"),
            bgcolor="rgba(255,87,34,0.1)",
            bordercolor="#FF5722",
            borderwidth=2
        )
    
    # Add END node
    fig.add_trace(go.Scatter(
        x=[positions["END"][0]], y=[positions["END"][1]],
        mode='markers+text',
        marker=dict(symbol='circle', size=60, color="#F44336"),
        text=["END"],
        textposition="middle center",
        name="END"
    ))
    
    # Add edges
    edges = [
        ("START", "Planner"),
        ("Planner", "Router"),
        ("Re-planner", "Router"),
        ("Re-planner", "Final Responder"),
        ("Final Responder", "END")
    ]
    
    for start, end in edges:
        if start == "Re-planner" and end == "Router":
            # Curved edge for loop back
            fig.add_trace(go.Scatter(
                x=[positions[start][0], positions[start][0], positions[end][0], positions[end][0]],
                y=[positions[start][1], 0.9, 0.9, positions[end][1]],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.5)', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[positions[start][0], positions[end][0]],
                y=[positions[start][1], positions[end][1]],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.5)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Router to Executors and back
    if st.session_state.executors:
        # To executors
        fig.add_trace(go.Scatter(
            x=[positions["Router"][0], positions["Executors"][0]],
            y=[positions["Router"][1], positions["Executors"][1]],
            mode='lines',
            line=dict(color='rgba(255,152,0,0.5)', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # From executors to Re-planner
        fig.add_trace(go.Scatter(
            x=[positions["Executors"][0], positions["Re-planner"][0]],
            y=[positions["Executors"][1], positions["Re-planner"][1]],
            mode='lines',
            line=dict(color='rgba(255,152,0,0.5)', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "🔬 Plan-Execute System Architecture",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        height=400,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig