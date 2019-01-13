class Tsetlin {

    constructor(numClasses, numClauses, numFeatures, numStates, s, threshold) {
        this.numClasses = numClasses;
        this.numClauses = numClauses;
        this.numFeatures = numFeatures;
        this.numStates = numStates;
        this.s = s;
        this.threshold = threshold;
        this.taState = [];
        this.initTaState();
        this.clauseSign = [];
        this.initClauseSigns();
        this.clauseCount = new Array(this.numClasses).fill(0);
        this.clauseOutput = new Array(this.numClauses).fill(0);
        this.classSum = new Array(this.numClasses).fill(0);
        this.feedBackToClauses = new Array(this.numClauses).fill(0);
        for (let i = 0; i < this.numClasses; i++) {
            for (let j = 0; j < Math.floor(this.numClauses / this.numClasses); j++) {
                this.clauseSign[i][this.clauseCount[i]][0] = i * (this.numClauses / this.numClasses) + j;
                if (j % 2 == 0)
                    this.clauseSign[i][this.clauseCount[i]][1] = 1;
                else
                    this.clauseSign[i][this.clauseCount[i]][1] = -1;
                this.clauseCount[i] += 1;
            }
        }
    }

    initClauseSigns() {
        for (let i = 0; i < this.numClasses; i++) {
            this.clauseSign[i] = [];
            for (let j = 0; j < this.numClauses / this.numClasses; j++) {
                this.clauseSign[i][j] = [];
                this.clauseSign[i][j][0] = 0;
                this.clauseSign[i][j][1] = 0;
            }
        }
    }

    initTaState() {
        for (let i = 0; i < this.numClauses; i++) {
            this.taState[i] = [];
            for (let j = 0; j < this.numFeatures; j++) {
                this.taState[i][j] = [];
                this.taState[i][j][0] = Math.floor(Math.random()) ? this.numStates : this.numStates + 1;
                this.taState[i][j][1] = Math.floor(Math.random()) ? this.numStates : this.numStates + 1;
            }
        }
    }

    action(state) {
        if (state <= this.numStates)
            return 0;
        else
            return 1;
    }

    calcClauseOutput(X, predict) {
        if (!predict || predict === "undefined")
            predict = 0;

        for (let j = 0; j < this.numClauses; j++) {
            this.clauseOutput[j] = 1;
            let allExclude = 1;
            for (let k = 0; k < this.numFeatures; k++) {
                let actionInclude = this.action(this.taState[j][k][0]);
                let actionIncludeNeg = this.action(this.taState[j][k][1]);
                if (actionInclude == 1 || actionIncludeNeg == 1)
                    allExclude = 0;
                if ((actionInclude == 1 && X[k] == 0) || (actionIncludeNeg == 1 && X[k] == 1)) {
                    this.clauseOutput[j] = 0;
                    break;
                }
            }
            if (predict == 1 && allExclude == 1)
                this.clauseOutput[j] = 0;
        }
    }

    sumUpClassVotes() {
        for (let targetClass = 0; targetClass < this.numClasses; targetClass++) {
            this.classSum[targetClass] = 0;
            for (let j = 0; j < this.clauseCount[targetClass]; j++) {
                this.classSum[targetClass] += this.clauseOutput[this.clauseSign[targetClass][j][0]] * this.clauseSign[targetClass][j][1];
                if (this.classSum[targetClass] > this.threshold)
                    this.classSum[targetClass] = this.threshold;
                else if (this.classSum[targetClass] < -this.threshold)
                    this.classSum[targetClass] = -this.threshold;
            }
        }
    }

    predict(X) {
        this.calcClauseOutput(X, 1);
        this.sumUpClassVotes();
        let maxClassSum = this.classSum[0];
        let maxClass = 0;
        for (let targetClass = 1; targetClass < this.numClasses; targetClass++) {
            if (maxClassSum < this.classSum[targetClass]) {
                maxClassSum = this.classSum[targetClass];
                maxClass = targetClass;
            }
        }
        return maxClass;
    }

    update(X, targetClass) {
        //Randomly pick one of the other classes, for pairwise learning of class output
        let negativeTargetClass = Math.floor(this.numClasses * (1.0 - 1e-15) * Math.random());
        while (negativeTargetClass == targetClass)
            negativeTargetClass = Math.floor(this.numClasses * (1.0 - 1e-15) * Math.random());
        //Calculate Clause Output
        this.calcClauseOutput(X);
        //sum up clause votes
        this.sumUpClassVotes();
        //calculate Feedback to Clauses
        for (let j = 0; j < this.numClauses; j++) // init feedback to clauses
            this.feedBackToClauses[j] = 0;

        for (let j = 0; j < this.clauseCount[targetClass]; j++) {
            if (Math.random() > (1.0 / this.threshold * 2) * (this.threshold - this.classSum[targetClass]))
                continue;

            if (this.clauseSign[targetClass][j][1] > 0)
                this.feedBackToClauses[this.clauseSign[targetClass][j][0]]++;
            else if (this.clauseSign[targetClass][j][1] < 0)
                this.feedBackToClauses[this.clauseSign[targetClass][j][0]]--;

        }
        for (let j = 0; j < this.clauseCount[negativeTargetClass]; j++) {
            if (Math.random() > (1.0 / this.threshold * 2) * (this.threshold + this.classSum[negativeTargetClass]))
                continue;

            if (this.clauseSign[negativeTargetClass][j][1] > 0)
                this.feedBackToClauses[this.clauseSign[negativeTargetClass][j][0]]--;
            else if (this.clauseSign[negativeTargetClass][j][1] < 0)
                this.feedBackToClauses[this.clauseSign[negativeTargetClass][j][0]]++;
        }


        //Train individual Automata
        for (let j = 0; j < this.numClauses; j++) {
            if (this.feedBackToClauses[j] > 0) {
                //Type I Feedback (Combats False Negatives)
                if (this.clauseOutput[j] == 0) {
                    for (let k = 0; k < this.numFeatures; k++) {
                        if (Math.random() <= 1.0 / this.s) {
                            if (this.taState[j][k][0] > 1)
                                this.taState[j][k][0]--;
                        }
                        if (Math.random() <= 1.0 / this.s) {
                            if (this.taState[j][k][1] > 1)
                                this.taState[j][k][1]--;
                        }

                    }
                } else if (this.clauseOutput[j] == 1) {
                    for (let k = 0; k < this.numFeatures; k++) {
                        if (X[k] == 1) {
                            if (Math.random() <= (this.s - 1) / this.s)
                                if (this.taState[j][k][0] < this.numStates * 2)
                                    this.taState[j][k][0]++;
                            if (Math.random() <= 1.0 / this.s)
                                if (this.taState[j][k][1] > 1)
                                    this.taState[j][k][1]--;
                        } else if (X[k] == 0) {
                            if (Math.random() <= (this.s - 1) / this.s)
                                if (this.taState[j][k][1] < this.numStates * 2)
                                    this.taState[j][k][1]++;
                            if (Math.random() <= 1.0 / this.s)
                                if (this.taState[j][k][0] > 1)
                                    this.taState[j][k][0]--;
                        }
                    }
                }
            } else if (this.feedBackToClauses[j] < 0) {
                // Type II Feedback (Combats False Positives
                if (this.clauseOutput[j] == 1) {
                    for (let k = 0; k < this.numFeatures; k++) {
                        let actionInclude = this.action(this.taState[j][k][0]);
                        let actionIncludeNegated = this.action(this.taState[j][k][1]);
                        if (X[k] == 0) {
                            if (actionInclude == 0 && this.taState[j][k][0] < this.numStates * 2)
                                this.taState[j][k][0]++;
                        } else if (X[k] == 1)
                            if (actionIncludeNegated == 0 && this.taState[j][k][1] < this.numStates * 2)
                                this.taState[j][k][1]++;
                    }
                }
            }
        }
    }

    evaluate(X, y) {
        let errors = 0;
        for (let l = 0; l < X.length; l++) {
            if (this.predict(X[l]) != y[l])
                errors++;
        }
        return 1.0 - errors / X.length;

    }
}

module.exports = Tsetlin;



