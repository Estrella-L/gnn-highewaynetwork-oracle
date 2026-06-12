import matplotlib.pyplot as plt
import networkx as nx

def load_grf(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    idx = 1

    # 节点信息
    node_ids = []
    node_labels = {}
    for _ in range(n):
        nid, lbl = lines[idx].split()
        nid = int(nid); lbl = int(lbl)
        node_ids.append(nid)
        node_labels[nid] = lbl
        idx += 1

    # 邻接信息（按节点顺序）
    edges = []
    for u in range(n):
        out_deg = int(lines[idx]); idx += 1
        for _ in range(out_deg):
            parts = lines[idx].split()
            v = int(parts[1])
            w = int(parts[2]) if len(parts) >= 3 else 1
            edges.append((u, v, w))
            idx += 1

    return node_ids, node_labels, edges

def main():
    grf_path = "cross30_full.grf"  # 改成你的文件
    node_ids, node_labels, edges = load_grf(grf_path)

    # 无向图可视化（去重）
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for u, v, w in edges:
        if u != v:
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=450)
    nx.draw_networkx_edges(G, pos, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(grf_path)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()