
function changeDataset(value) {

    var format = value; // Get the data set's format.

    if (format == "empty") { // This is a function that handles a new file, which users can upload.
      d3.select("#data").select("input").remove();
      $("#data").html(''); // Print on the screen the classification label.
      d3.select("#data")
        .append("input")
         .attr("type", "file")
         .style("font-size", 'calc(0.35em + 0.9vmin)')
         .on("change", function() {
          var file = d3.event.target.files[0];
          getfile(file);
          $("#data").html(file.name);
        })
    } else {
      $("#data").html('Datasets'); // Print on the screen the classification label.
      d3.select("#data").select("input").remove(); // Remove the selection field.
    }

}
