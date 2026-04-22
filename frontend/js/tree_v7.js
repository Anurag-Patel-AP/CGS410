// Utility for drawing trees
class TreeVisualizer {
    constructor(containerId, color) {
        this.containerId = containerId;
        this.color = color || "#2E86AB";
    }
    
    initSvg() {
        const el = document.getElementById(this.containerId);
        this.width = el.clientWidth || 800;
        this.height = el.clientHeight || 500;
        el.innerHTML = '';
        
        this.svg = d3.select("#" + this.containerId)
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%");
            
        this.g = this.svg.append("g").attr("transform", `translate(${this.width / 2}, 60)`);

        this.svg.call(d3.zoom().on("zoom", (event) => {
            this.g.attr("transform", `translate(${event.transform.x + this.width / 2}, ${event.transform.y + 60}) scale(${event.transform.k})`);
        }));
            
        // Build sleek, modern chevron arrowhead
        this.svg.append("defs").append("marker")
            .attr("id", "arrowhead-" + this.containerId)
            .attr("viewBox", "-2 -5 12 10")
            .attr("refX", 8) 
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 7)
            .attr("markerHeight", 7)
            .attr("xoverflow", "visible")
            .append("svg:path")
            .attr("d", "M 0,-4 L 8,0 L 0,4")
            .attr("fill", "none")
            .attr("stroke", "rgba(255, 255, 255, 0.5)")
            .attr("stroke-width", "1.5px")
            .attr("stroke-linecap", "round")
            .attr("stroke-linejoin", "round");
    }

    draw(tokens, treeEdges, deprels = null, goldMap = null) {
        this.initSvg();
        
        const n = tokens.length;
        // Build nodes
        const nodes = Array.from({length: n}, (_, i) => ({
            id: i,
            name: tokens[i],
            children: [],
            deprel: deprels ? deprels[i] : ""
        }));
        
        // Find roots and build hierarchy
        let roots = [];
        for (let i = 0; i < n; i++) {
            const parentIndex = treeEdges[i];
            if (parentIndex === -1 || parentIndex === undefined || parentIndex === null) {
                roots.push(nodes[i]);
            } else {
                nodes[parentIndex].children.push(nodes[i]);
            }
        }
        
        let rootData;
        if (roots.length === 1) {
            rootData = roots[0];
        } else {
            rootData = { id: "root", name: "ROOT", children: roots, deprel: "" };
        }
        
        const root = d3.hierarchy(rootData);
        
        // Calculate layout
        // Use nodeSize instead of fixed size to allow infinite horizontal expansion without overlap
        const treeLayout = d3.tree()
            .nodeSize([120, 90])
            .separation((a, b) => a.parent === b.parent ? 1 : 1.2);
        
        treeLayout(root);
        
        // Links
        const linkGenerator = d3.linkVertical()
            // start slightly below parent text
            .source(d => [d.source.x, d.source.y + 15])
            // end slightly above child text
            .target(d => [d.target.x, d.target.y - 15]);
            
        // Provide standard and green/red variants for the arrowhead definition dynamically on link call
        // we defined arrowhead-containerId in initSvg as white "rgba(255, 255, 255, 0.5)".
        // add multiple definitions in initSvg or just define them now if needed.
        // we can define them on the fly here to keep code clean.
        this.svg.select("defs").append("marker")
            .attr("id", "arrow-correct-" + this.containerId)
            .attr("viewBox", "-2 -5 12 10")
            .attr("refX", 8) 
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 7)
            .attr("markerHeight", 7)
            .append("svg:path")
            .attr("d", "M 0,-4 L 8,0 L 0,4")
            .attr("fill", "none")
            .attr("stroke", "var(--accent-green, #10b981)")
            .attr("stroke-width", "2px");
            
        this.svg.select("defs").append("marker")
            .attr("id", "arrow-incorrect-" + this.containerId)
            .attr("viewBox", "-2 -5 12 10")
            .attr("refX", 8) 
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 7)
            .attr("markerHeight", 7)
            .append("svg:path")
            .attr("d", "M 0,-4 L 8,0 L 0,4")
            .attr("fill", "none")
            .attr("stroke", "var(--accent-red, #ef4444)")
            .attr("stroke-width", "2px");

        this.g.selectAll(".link")
            .data(root.links())
            .enter().append("path")
            .attr("class", "link")
            .attr("d", linkGenerator)
            .attr("marker-end", d => {
                if (!goldMap || d.target.data.id === "root" || d.source.data.id === "root") return `url(#arrowhead-${this.containerId})`;
                const i = d.target.data.id;
                const predicted_p = treeEdges[i];
                const gold_p = goldMap[i];
                return predicted_p === gold_p ? `url(#arrow-correct-${this.containerId})` : `url(#arrow-incorrect-${this.containerId})`;
            })
            .attr("fill", "none")
            .attr("stroke", d => {
                if (!goldMap || d.target.data.id === "root" || d.source.data.id === "root") return "rgba(255, 255, 255, 0.3)";
                const i = d.target.data.id;
                const predicted_p = treeEdges[i];
                const gold_p = goldMap[i];
                // Make edges correctly colored
                return predicted_p === gold_p ? "var(--accent-green, #10b981)" : "var(--accent-red, #ef4444)";
            })
            .attr("stroke-width", d => {
                if (!goldMap || d.target.data.id === "root" || d.source.data.id === "root") return "1.5px";
                const i = d.target.data.id;
                return treeEdges[i] === goldMap[i] ? "2.5px" : "2px";
            });
            
        // Link Labels
        this.g.selectAll(".link-label")
            .data(root.links())
            .enter().append("text")
            .attr("class", "link-label")
            .attr("x", d => (d.source.x + d.target.x) / 2 + 5)
            .attr("y", d => (d.source.y + d.target.y) / 2 - 5)
            .text(d => d.target.data.deprel || "")
            .attr("fill", "var(--accent-blue)")
            .attr("font-size", "11px")
            .attr("font-weight", "600")
            .attr("text-shadow", "0 2px 4px rgba(0,0,0,1)");

        // Nodes
        const node = this.g.selectAll(".node")
            .data(root.descendants())
            .enter().append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        node.append("text")
            .attr("dy", 5)
            .attr("text-anchor", "middle")
            .text(d => d.data.name)
            .attr("fill", "#ffffff")
            .attr("font-size", "15px")
            .attr("font-weight", "500")
            .attr("text-shadow", "0 2px 4px rgba(0,0,0,0.9)");
    }
}
