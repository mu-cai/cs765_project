<script>
    var canvas = document.getElementById("tSNE_canvas");
    var ctx = canvas.getContext("2d");

    ctx.beginPath();
    for (var i = 0; i < 10000; i++){
        var cx = Math.floor(Math.random() * canvas.width);
        var cy = Math.floor(Math.random() * canvas.height);
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, 1, 0, Math.PI*2);
    }
    
    ctx.fillStyle = "#26963c";
    ctx.fill();
  </script>


<script>
    var nodes = [];
    d3.csv("mnist_sample_per_class_50_num_neighbors_3.csv", function(error, data){
      if (error) throw error;
      for (var i = 0; i < data.length; i++){
        nodes.push({index: i, 
          size: 50, 
          high_nei: data[i].high_nei, 
          high_sim: data[i].high_sim, 
          low_nei: data[i].low_nei, 
          low_sim: data[i].low_sim});
      }
    });
    data = {
          nodes: [
              {size: 50, value: 1},
              {size: 10, value: 2},
              {size: 10, value: 3},
              {size: 10, value: 4}
          ],
          links: [
              {center: 0, nei: [1, 2], nei_2: [1, 3]},
              {center: 1, nei: [3, 0], nei_2: [2, 3]},
              {center: 2, nei: [1, 3], nei_2: [0, 1]},
              {center: 3, nei: [0, 1], nei_2: [1, 2]}
          ]
      }
      var nodes = data.nodes;
      var links = data.links;
      var neiByindex = {};
      links.forEach(function(d){
          neiByindex[d.center] = d.nei;
      });
      var neiiByindex = {};
      links.forEach(function(d){
          neiiByindex[d.center] = d.nei_2;
      })
    
    var mouseOverfunc = function(d){
        var circle = d3.select(this);

        div
        .transition(500)
        .style("opacity", .9);
        div
        .html(d.size + "</br>" + d.value + "</br>" + "<img src='http://marvel-force-chart.surge.sh/marvel_force_chart_img/marvel.png' width=100% height=100%/>")
        .style("left", (d3.event.pageX) + "px")		
        .style("top", (d3.event.pageY - 28) + "px");

        node
        .transition(500)
        .style("opacity", function(o){
            return ishighTopk(o, d) || islowTopk(o, d) || isEqual(o, d) ? 1.0 : 0.2;
        })
        .style("fill", function(o){
            if (ishighTopk(o, d) && islowTopk(o, d)){
                fillcolor = "red";
            } else if (ishighTopk(o, d)){
                fillcolor = "green";
            } else if (islowTopk(o, d)){
                fillcolor = "blue";
            } else if(isEqual(o, d)){
                fillcolor = "hotpink";
            } else {
                fillcolor = "#000";
            }
            let high = "<img src='http://placekitten.com/200/200' />" + "<img src='http://placekitten.com/200/200' />";
            tSNEhigh.transition(500).style("opcacity", 1);
            tSNEhigh.html(high);
            return fillcolor;
        });

        circle
        .transition(500)
        .attr("r", function(){
            return 1.4 * node_radius(d);
        });
    }

    var mouseOutfunc = function(){
        var circle = d3.select(this);

        node.transition(500);
        circle.transition(500)
        .attr("r", node_radius);

        div.transition(500).style("opacity", 0);
    }

    function isEqual(a, b){
        return a.index == b.index;
    }

    function ishighTopk(a, b){
        return neiByindex[b.index].includes(a.index);
    }

    function islowTopk(a, b){
        return neiiByindex[b.index].includes(a.index);
    }

    function node_radius(d){
        return Math.pow(40.0 * d.size, 1/3);
    }

    function tick() {
    node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
    }

    var width = 1000;
    var height = 800;

    var force = d3.layout.force()
    .nodes(nodes)
    .charge(-3000)
    .friction(0.6)
    .gravity(0.6)
    .size([width,height])
    .start();

    var div = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

    var tSNEhigh = d3.select("#tSNEhigh").append("tSNEhigh");

    var svg = d3.select("#tSNEcanvas").append("svg")
    .attr("width", width)
    .attr("height", height);

    var node = svg.selectAll(".node")
    .data(nodes)
    .enter().append("g")
    .attr("class", "node")
    .call(force.drag);

    node
    .append("circle")
    .attr("r", node_radius)
    .on("mouseover", mouseOverfunc)
    .on("mouseout", mouseOutfunc);
    
    
    force
    .on("tick",tick);
</script>