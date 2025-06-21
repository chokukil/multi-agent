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
    """Plan-Execute 구조를 시각화 (최적화된 역할 반영)"""
    if not VISUALIZATION_AVAILABLE:
        st.warning("📊 Visualization packages not installed. System structure:")
        
        # Text-based visualization
        st.markdown("""
        ```
        START → Planner → Router → Specialized Executors → Re-planner → Router
                            ↑__________________________________|
                                                                ↓
                                                        Final Responder → END
        ```
        """)
        
        if st.session_state.executors:
            # 새로운 역할들 표시
            executor_names = list(st.session_state.executors.keys())
            st.info(f"🤖 **Specialized Executors** ({len(executor_names)}): {', '.join(executor_names[:3])}...")
            
            # 역할별 전문성 간단 표시
            st.markdown("**🎯 Specialized Roles:**")
            role_mapping = {
                "Data_Validator": "🔍 Data Quality & Validation",
                "Preprocessing_Expert": "🛠️ Data Preprocessing & Feature Engineering", 
                "EDA_Analyst": "📊 Exploratory Data Analysis",
                "Visualization_Expert": "📈 Data Visualization & Communication",
                "ML_Specialist": "🤖 Machine Learning & Modeling",
                "Statistical_Analyst": "📈 Statistical Analysis & Testing",
                "Report_Generator": "📄 Professional Reporting & Documentation"
            }
            
            for executor_name in executor_names:
                role_desc = role_mapping.get(executor_name, "🔧 Data Science Expert")
                st.write(f"• **{executor_name}**: {role_desc}")
        else:
            st.warning("⚠️ **Executors**: None - Please create executors")
        return None
    
    # Create figure when visualization is available
    fig = go.Figure()
    
    # Node positions (더 넓은 배치로 전문성 강조)
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
        marker=dict(symbol='circle', size=70, color="#4CAF50", line=dict(width=2, color="#2E7D32")),
        text=["🚀 START"],
        textposition="middle center",
        textfont=dict(size=12, color="white"),
        name="START",
        hovertemplate="<b>System Start</b><br>Begin analysis pipeline<extra></extra>"
    ))
    
    # Add system nodes with enhanced styling
    system_nodes = [
        ("Planner", "🎯 PLANNER", "#2196F3", "Creates execution strategy"),
        ("Router", "🔀 ROUTER", "#FF9800", "Routes to specialized experts"), 
        ("Re-planner", "🔄 RE-PLANNER", "#9C27B0", "Evaluates progress & adapts"),
        ("Final Responder", "📋 FINALIZER", "#607D8B", "Synthesizes results")
    ]
    
    for node_name, display_name, color, description in system_nodes:
        fig.add_trace(go.Scatter(
            x=[positions[node_name][0]], y=[positions[node_name][1]],
            mode='markers+text',
            marker=dict(symbol='circle', size=65, color=color, line=dict(width=2, color="white")),
            text=[display_name],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name=node_name,
            hovertemplate=f"<b>{display_name}</b><br>{description}<extra></extra>"
        ))
    
    # Add specialized executors node with enhanced info
    if st.session_state.executors:
        executor_count = len(st.session_state.executors)
        executor_names = list(st.session_state.executors.keys())
        
        # Executor 노드 - 더 큰 크기로 전문성 강조
        fig.add_trace(go.Scatter(
            x=[positions["Executors"][0]], y=[positions["Executors"][1]],
            mode='markers+text',
            marker=dict(symbol='hexagon', size=80, color="#E91E63", line=dict(width=3, color="white")),
            text=[f"🧠 EXPERTS<br>({executor_count})"],
            textposition="middle center",
            textfont=dict(size=11, color="white"),
            name="Specialized Executors",
            hovertemplate=f"<b>🧠 Specialized Data Science Experts</b><br>" +
                         f"Total: {executor_count} experts<br>" +
                         f"Roles: {', '.join(executor_names[:3])}" +
                         ("..." if len(executor_names) > 3 else "") +
                         "<extra></extra>"
        ))
        
        # Executor 전문성을 보여주는 작은 노드들 추가
        role_positions = [
            (2.7, 0.8), (2.7, 0.6), (2.7, 0.4), (2.7, 0.2),  # 왼쪽
            (3.3, 0.8), (3.3, 0.6), (3.3, 0.4)              # 오른쪽
        ]
        
        role_colors = ["#FF5722", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#795548", "#607D8B"]
        role_symbols = ["🔍", "🛠️", "📊", "📈", "🤖", "📈", "📄"]
        
        for i, (executor_name, (x, y)) in enumerate(zip(executor_names[:7], role_positions)):
            if i < len(role_colors):
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(symbol='circle', size=25, color=role_colors[i], 
                               line=dict(width=1, color="white")),
                    text=[role_symbols[i]],
                    textposition="middle center",
                    textfont=dict(size=8),
                    name=executor_name,
                    hovertemplate=f"<b>{executor_name}</b><br>Specialized Expert<extra></extra>",
                    showlegend=False
                ))
    else:
        # No executors - 설정 필요 표시
        fig.add_trace(go.Scatter(
            x=[positions["Executors"][0]], y=[positions["Executors"][1]],
            mode='markers+text',
            marker=dict(symbol='circle', size=70, color="#BDBDBD", line=dict(width=2, color="#757575")),
            text=["⚙️ SETUP<br>NEEDED"],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            name="Setup Required",
            hovertemplate="<b>No Executors Configured</b><br>Please create specialized experts<extra></extra>"
        ))
    
    # Add END node
    fig.add_trace(go.Scatter(
        x=[positions["END"][0]], y=[positions["END"][1]],
        mode='markers+text',
        marker=dict(symbol='circle', size=70, color="#F44336", line=dict(width=2, color="#C62828")),
        text=["🏁 END"],
        textposition="middle center",
        textfont=dict(size=12, color="white"),
        name="END",
        hovertemplate="<b>Analysis Complete</b><br>Results ready for review<extra></extra>"
    ))
    
    # Add edges with enhanced styling
    edges = [
        ("START", "Planner", "#4CAF50", 3),
        ("Planner", "Router", "#2196F3", 3),
        ("Re-planner", "Final Responder", "#9C27B0", 3),
        ("Final Responder", "END", "#607D8B", 3)
    ]
    
    for start, end, color, width in edges:
        fig.add_trace(go.Scatter(
            x=[positions[start][0], positions[end][0]],
            y=[positions[start][1], positions[end][1]],
            mode='lines',
            line=dict(color=color, width=width),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Special curved edge for re-planner loop back
    fig.add_trace(go.Scatter(
        x=[positions["Re-planner"][0], positions["Re-planner"][0], positions["Router"][0], positions["Router"][0]],
        y=[positions["Re-planner"][1], 0.9, 0.9, positions["Router"][1]],
        mode='lines',
        line=dict(color='#9C27B0', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Router to Executors and back (enhanced for multiple executors)
    if st.session_state.executors:
        # To executors - 더 굵은 선으로 중요성 강조
        fig.add_trace(go.Scatter(
            x=[positions["Router"][0], positions["Executors"][0]],
            y=[positions["Router"][1], positions["Executors"][1]],
            mode='lines',
            line=dict(color='#E91E63', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # From executors to Re-planner
        fig.add_trace(go.Scatter(
            x=[positions["Executors"][0], positions["Re-planner"][0]],
            y=[positions["Executors"][1], positions["Re-planner"][1]],
            mode='lines',
            line=dict(color='#E91E63', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout with enhanced styling
    fig.update_layout(
        title={
            'text': "🔬 Optimized Plan-Execute Data Science Architecture",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=22, color="#1565C0")
        },
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        height=450,
        hovermode='closest',
        plot_bgcolor='rgba(245,245,245,0.8)',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=80, b=20),
        annotations=[
            dict(
                text="🎯 Specialized Expert Workflow",
                x=0.5, y=0.02,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color="#666")
            )
        ]
    )
    
    return fig