$(document).ready(function () {

    var loadFile = function (filePath, done) {
        var xhr = new XMLHttpRequest();
        xhr.onload = function () { return done(this.responseText) };
        xhr.open("GET", filePath, false);
        xhr.send();
    };

    // paths to all of your files
    var jsonFiles = [ "static/nnet_outputs/nnet_output_iter", "static/nnet_outputs/nnet_output_v1" ];
    var nnetData = [];

    jsonFiles.forEach(function (file, i) {
        loadFile(file, function (responseText) {
            nnetData[i] = JSON.parse(responseText);
        });
    });

    loadDataIntoGraphs(nnetData);



    // loadJSON('static/nnet_outputs/nnet_output_v1', function(text) {
    //     let data = JSON.parse(text);
    //    // console.log(new Array(30).fill(data["test_sizes"]));
    //
    //      var labels = {
    //         title: "Matrix Based: Percent Correct over 30 Epochs",
    //         xaxis: "Epochs",
    //         yaxis: "% Correct"
    //     };
    //
    //     errorOverEpochGraph("percent-error-mat", data["test_results"], data["test_sizes"], labels);
    // });
    //
    // loadJSON('static/nnet_outputs/nnet_output_iter', function(text) {
    //     let data = JSON.parse(text);
    //    // console.log(new Array(30).fill(data["test_sizes"]));
    //
    //     var labels = {
    //         title: "Iterative: Percent Correct over 30 Epochs",
    //         xaxis: "Epochs",
    //         yaxis: "% Correct"
    //     };
    //
    //     errorOverEpochGraph("percent-error-iter", data["test_results"], data["test_sizes"], labels);
    // });
});

function loadDataIntoGraphs(dataSets) {
    var iter_nnet = dataSets[0];
    var mat_nnet = dataSets[1];


    var labels = {
        title: "Iterative vs Matrix Based: Percent Correct over 30 Epochs",
        xaxis: "Epochs",
        yaxis: "% Correct",
        trace1: "Iterative",
        trace2: "Matrix-based"
    };

    errorOverEpochGraph(
        "percent-error-graph",
        iter_nnet["test_results"],
        mat_nnet["test_results"],
        iter_nnet["test_sizes"],
        labels
    );

    epochOverTime(
        "",
        iter_nnet["epoch_delta_times"],
        mat_nnet["epoch_delta_times"],
        labels
    );
}

function epochOverTime(div, timeDeltas1, timeDeltas2, labels) {
    let epochIndex = Array(timeDeltas1.length).fill().map((x, index) => index);

    var sum = 0;
    let timeProgression = timeDeltas1.map(elem => sum = (sum || 0) + elem);
    console.log(timeProgression);
    console.log(timeDeltas1);
}


function errorOverEpochGraph(div, testResults1, testResults2, testSize, labels) {
    let error1 = testResults1.map(x => x / testSize);
    let error2 = testResults2.map(x => x / testSize);
    let epochIndex = Array(testResults1.length).fill().map((x, index) => index);

    var trace1 = {
        x: epochIndex,
        y: error1,
        name: labels["trace1"],
        type: 'scatter'
    };

    var trace2 = {
        x: epochIndex,
        y: error2,
        name: labels["trace2"],
        type: 'scatter'
    };

    var layout = {
        title: {
            text: labels["title"],
            font: {
              family: 'Courier New, monospace',
              size: 12
            },
            xref: 'paper',
            x: 0.00,
          },
          xaxis: {
            title: {
              text: labels["xaxis"],
              font: {
                family: 'Courier New, monospace',
                size: 14,
                color: '#7f7f7f'
              }
            },
          },
          yaxis: {
            title: {
              text: labels["yaxis"],
              font: {
                family: 'Courier New, monospace',
                size: 14,
                color: '#7f7f7f'
              }
            }
          }
    };

    Plotly.newPlot(div, [trace1, trace2], layout);
}


function loadData() {
    var deferred = $.Deferred();

    var loadFile = function (filePath, done) {
        var xhr = new XMLHttpRequest();
        xhr.onload = function () { return done(this.responseText) };
        xhr.open("GET", filePath, true);
        xhr.send();
    };

      // paths to all of your files
    var jsonFiles = [ "static/nnet_outputs/nnet_output_iter", "static/nnet_outputs/nnet_output_v1" ];
    var nnetData = [];

    jsonFiles.forEach(function (file, i) {
        loadFile(file, function (responseText) {
            nnetData[i] = JSON.parse(responseText);

            // TODO use 1 bc 2 files are loaded in, resolve after last file loads
            if (i === 1) {
                deferred.resolve();
            }
        });
    });

    return deferred;
}

function loadJSON(file, callback) {

    // let req = new XMLHttpRequest();
    //
    // req.overrideMimeType("application/json");
    // req.open('GET', file, true);
    //
    // req.onreadystatechange = function () {
    //
    //       if (req.readyState == 4 && req.status == "200") {
    //         // requires callback bc asychronous
    //         callback(req.responseText);
    //       }
    // };
    //
    // req.send(null);

    // var loadFile = function (filePath, done) {
    //     var xhr = new XMLHttpRequest();
    //     xhr.onload = function () { return done(this.responseText) }
    //     xhr.open("GET", filePath, true);
    //     xhr.send();
    // };
    //
    // // paths to all of your files
    // var jsonFiles = [ "static/nnet_outputs/nnet_output_iter", "static/nnet_outputs/nnet_output_v1" ];
    // var jsonData = [];
    //
    // jsonFiles.forEach(function (file, i) {
    //     loadFile(file, function (responseText) {
    //         jsonData[i] = JSON.parse(responseText);
    //     });
    // });
 }