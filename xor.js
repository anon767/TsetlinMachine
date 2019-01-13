const Tsetlin = require("./tsetlin.js");
const fs = require('fs');
function loadTrainData(callback) {
    fs.readFile("NoisyXORTestData.txt", 'utf8', callback);
}

function evaluateXOR(response) {
    let tsetlin = new Tsetlin(2, 20, 12, 100, 3.9, 15);
    let samples = response.split(/\r?\n/);
    for (let i = 0; i < 200; i++) {
        samples.forEach((sample) => { //online training
            if (sample.length <= 0)
                return;
            let tmpsample = sample.split(" ").map((val) => parseInt(val.trim()));
            let y = tmpsample[12];
            delete tmpsample[12];
            tsetlin.update(tmpsample, y);
        });
        if (i % 10 == 0)
            console.log(i, "epoch");

    }
    console.log("Prediction X1 = 1, X2 = 0, ... -> y = ", tsetlin.predict([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]))

    console.log("Prediction X1 = 1, X2 = 1, ... -> y = ", tsetlin.predict([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]))

    console.log("Prediction X1 = 0, X2 = 0, ... -> y = ", tsetlin.predict([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]))

    console.log("Prediction X1 = 0, X2 = 1, ... -> y = ", tsetlin.predict([0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]))

    X = samples.map(sample => sample.split(" ").filter((val, idx) => (idx < 12)).map(val => parseInt(val.trim())));
    y = samples.map(sample => parseInt(sample.split(" ")[12]));
    console.log("accuracy ", tsetlin.evaluate(X, y));
}

loadTrainData((err, response) => {
    console.log("XOR training set without Noise");
    evaluateXOR(response);
})

