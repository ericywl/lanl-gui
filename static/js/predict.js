window.onload = function () {
    document.getElementById("files").onchange = function () {
        // Read the image file as a data URL.
        types = ["text/csv"];
        // Check file type is JPG image
        if (types.indexOf(this.files[0].type) < 0) {
            console.log("File must be CSV!");
            return;
        }
        // Get loaded data and render thumbnail.
        document.getElementById("files-label").innerText = this.files[0].name;
        document.getElementById("predict-btn").disabled = false;
    };

    document.getElementById("predict-btn").onclick = function (e) {
        e.preventDefault()
        file = document.getElementById("files").files[0];
        const reader = new FileReader();
        reader.onload = function (e) {
            const csvBlob = e.target.result;
            const json = { name: file.name, data: csvBlob };
            makePostRequest("/predict", json)
                .then(function (resp) {
                    output = JSON.parse(resp.response)
                    console.log(output)
                    document.getElementById("files").disabled = false;
                    document.getElementById("image").src = "static/img/preds/" + output.filename
                })
                .catch(function (error) {
                    console.log(error);
                });
        };
        reader.readAsDataURL(file)
        document.getElementById("files").disabled = true;
    };
};

const makePostRequest = function (url, json) {
    const xhr = new XMLHttpRequest();
    return new Promise(function (resolve, reject) {
        xhr.onreadystatechange = function () {
            // Request not ready
            if (xhr.readyState !== 4) {
                document.getElementById("predict-btn").disabled = true;
                return;
            }
            // Process response
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(xhr);
            } else {
                reject({
                    status: xhr.status,
                    statusText: xhr.statusText
                });
            }
        };

        xhr.open("POST", url, true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.send(JSON.stringify(json));
    });
};

const json2Table = function (json) {
    const orderedJsonArray = Object.keys(json)
        .map(function (key) {
            return [key, json[key]];
        })
        .sort(function (a, b) {
            return b[1] - a[1];
        });

    tableStr =
        "<table class='table' style='margin-top: 10px'>" +
        "<thead><tr>" +
        "<th scope='col'>#</th>" +
        "<th scope='col'>Label</th>" +
        "<th scope='col'>Confidence</th>" +
        "</tr></thead>" +
        "<tbody>";

    for (let i = 0, len = orderedJsonArray.length; i < len; i++) {
        tmp = "<tr><th scope='row'>";
        tmp += (i + 1);
        tmp += "</th>";
        tmp += "<td>";
        tmp += orderedJsonArray[i][0];
        tmp += "</td>";
        tmp += "<td>";
        tmp += (orderedJsonArray[i][1] * 100).toFixed(2) + '%';
        tmp += "</td>";
        tmp += "</tr>";
        tableStr += tmp;
    }
    tableStr += "</tbody></table>";
    return tableStr
};
