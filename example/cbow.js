
const epoch = 100;
const window = 5;
const dim = 50;
const learningRate = 0.01;

var wordWeights = [];
var vocabulary = [];

function lrnStart(str){

    let words = str.split(' ');
    vocabulary = Array.from(new Set(words));
    wordWeights = initWord(vocabulary);
    
    // console.log(`wordWeights ${wordWeights}`)

    // 학습
    for(let i=0; i<epoch; i++){
        for(let wordIndex = 0; wordIndex < words.length; wordIndex++){
            let w = words[wordIndex];
            let inputDatas = words.slice(Math.max(0, wordIndex - window), wordIndex)
                                .concat(words.slice(wordIndex + 1, wordIndex + window + 1));

            // context strings avg > 1 * dim 행렬 결과
            // 평균벡터 혹은 히든벡터, 투사층이라 불리며 주변단어 벡터의 평균 연산
            let hiddenLayerAvg = wordVectorAvg(inputDatas, wordWeights, vocabulary);

            // 1 * word length 행렬 결과
            // 중심단어 예측 결과이며 가중치 행렬의 곱
            let predicted = predict(hiddenLayerAvg, wordWeights);

            // words 0 ~ 1 softmax >  1 * word length 행렬
            let softmaxed = softmax(predicted);

            // 손실률 계산
            let loss = crossEntropy(softmaxed, vocabulary.indexOf(w));

            // 1 X word length 행렬결과
            let gradients = gradient(softmaxed, vocabulary.indexOf(w));

            // 파라미터 업데이트
            updateWeights(wordWeights, hiddenLayerAvg, gradients, learningRate)
        }
    }

    // console.log(`wordWeights ${wordWeights}`)

}

// 가중치 업데이트( 모델 학습 )
function updateWeights(weights, hiddenLayerAvg, gradients, learningRate) {
    weights.forEach((weight, index) => {
        weight.forEach((_, dimIndex) => {
            weights[index][dimIndex] -= learningRate * gradients[index] * hiddenLayerAvg[dimIndex];
        });
    });
}

// 가중치 행렬(W, W') 조정값
function gradient(softmaxed, index) {
    return softmaxed.map((value, i) => value - (i === index ? 1 : 0));
}

// 학습 데이터 정확도, 손실값
function crossEntropy(softmaxed, index) {
    return -Math.log(softmaxed[index] + Number.EPSILON);
}

// 0~1 사이의 정규화
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
    const sumExpLogits = expLogits.reduce((acc, curr) => acc + curr, 0);
    const softmaxProbs = expLogits.map(expLogit => expLogit / sumExpLogits);
    return softmaxProbs;
}

// 평균(행렬)값 * W` = 중심단어 예측
function predict(avg, wordWeights){
    return wordWeights.map(weight => weight.reduce((acc, w, index) => acc + w * avg[index], 0));
}

// 주변단어 임베딩 평균 구하기
function wordVectorAvg(context, weights, vocabulary) {
    let sum = new Array(dim).fill(0);
    context.forEach(word => {
        let wordIndex = vocabulary.indexOf(word);
        weights[wordIndex].forEach((w, index) => {
            sum[index] += w;
        });
    });
    return sum.map(value => value / context.length);
}

function documentVectorAvg(document, weights, vocabulary) {
    // 문서를 단어로 분리
    const words = document.split(/\s+/);
    const docVector = new Array(dim).fill(0);
    let validWords = 0;

    words.forEach(word => {
        let wordIndex = vocabulary.indexOf(word);
        if(wordIndex > -1){
            validWords++;
            // 임베딩이 있는 단어만 고려
            const wordEmbedding = weights[wordIndex];
            wordEmbedding.forEach((value, index) => {
                docVector[index] += value; // 해당 차원의 값을 누적
            });
        }
    });

    if (validWords > 0) {
        // 평균을 계산하여 문서 벡터 반환
        return docVector.map(value => value / validWords);
    } else {
        // 유효한 단어가 없는 경우 0 벡터 반환
        return docVector;
    }
}

// 임베딩 랜덤(0~1)으로 초기화
function initWord(words){
    const vocabularySize = words.length;
    const wordEmbeddings = [];
    for (let w = 0; w < vocabularySize; w++) {
        const embedding = [];
        for (let d = 0; d < dim; d++) {
            embedding.push(Math.random());
        }
        wordEmbeddings[w] = embedding;
    }

    return wordEmbeddings;
}


// ====================================== 테스트 ======================================================

// 코사인 유사도 계산 함수
function cosineSimilarity(vecA, vecB) {

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function wordSimilarity(lrnText, testWord){

    lrnStart(lrnText);
    
    let maxSimilarity = -Infinity;
    let maxSimilarIndex = 0;
    let mostSimilarWord = '';
    let trgetIndex = vocabulary.indexOf(testWord);
    console.log(`테스트 단어! ::: ${vocabulary[trgetIndex]} / 위치 ::: ${trgetIndex}`);
    for (var w=0; w<vocabulary.length; w++) {
        if (w !== trgetIndex) {
            const similarity = cosineSimilarity(wordWeights[w], wordWeights[trgetIndex]);
            // console.log(`similarWord ${vocabulary[w]}`);
            // console.log(`similarity ${similarity}`);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                mostSimilarWord = vocabulary[w];
                maxSimilarIndex = w;
            }
        }
    }
    console.log(`유사 단어 : ${mostSimilarWord} / 유사 단어 위치 : ${maxSimilarIndex} / 유사도 : ${maxSimilarity}`);

}

function docSimilarity(lrnText, testDoc) {

    lrnStart(lrnText);

    const sentences = testDoc.split('.');  // 문장 분리
    const docVector = documentVectorAvg(testDoc, wordWeights, vocabulary);  // 전체 문서 벡터
    const summaries = [];

    sentences.forEach(sentence => {
        if(sentence.trim() == '') return;
        const sentenceVector = documentVectorAvg(sentence, wordWeights, vocabulary);  // 각 문장 벡터
        const similarity = cosineSimilarity(docVector, sentenceVector);  // 코사인 유사도 계산
        console.log(`similarity :: ${similarity} // sentence :: ${sentence}`);
        if (similarity > 0.5) {  // 설정된 임계값(0.5) 이상의 유사도를 가진 문장만을 요약으로 사용합니다
            summaries.push(sentence);
        }
    });

    console.log(`result :: ${summaries.join(', ')}`);

}

let lrnWord = `CBOW 모델을 사용할 때, 전체 문장 집합으로부터 단어의 길이나 어휘 사전(Vocabulary)의 크기를 정하는 것은 학습 과정에서 중요한 요소입니다. 
여기서 "단어의 길이"라는 말이 임베딩 벡터의 차원을 의미하는 것이라면, 이는 일반적으로 모델 설계 단계에서 결정되는 값입니다. 하지만 
"어휘 사전의 크기"는 훈련 데이터에 등장하는 모든 유니크한 단어들을 기반으로 설정되며, CBOW 모델의 입력 및 출력층의 크기를 결정하는 데 사용됩니다.
어휘 사전의 크기 설정 어휘 사전의 크기는 모든 문장에서 추출한 유니크한 단어들의 수에 따라 결정됩니다. 이 사전은 모델이 처리할 수 있는 모든 단어를 포함하며, 
각 단어는 원-핫 인코딩으로 표현됩니다. 예를 들어, 어휘 사전의 크기가 10,000이라면, 입력층과 출력층의 노드 수도 각각 10,000개가 됩니다.
임베딩 벡터의 차원 임베딩 벡터의 차원은 모델이 각 단어를 표현하는데 사용하는 속성의 수를 의미합니다. 이 차원 수는 실험을 통해 결정될 수 있으며, 
일반적으로는 50, 100, 200, 300 등의 값을 사용합니다. 차원 수는 모델의 복잡성과 성능에 영향을 미칩니다: 차원 수가 높을수록 더 많은 정보를 담을 수 있지만, 계산 비용도 증가하고, 과적합(overfitting)의 위험도 커질 수 있습니다.
예시예를 들어, 여러 문장에서 어휘를 추출하고 이를 기반으로 CBOW 모델을 훈련시키는 과정은 다음과 같습니다.
데이터 준비: 모든 문장을 토큰화하여 유니크한 단어들을 추출합니다.
어휘 사전 구축: 추출된 유니크한 단어들로부터 어휘 사전을 만듭니다. 각 단어는 사전에서 고유한 인덱스를 가집니다.
입력 및 타겟 데이터 생성: 각 문장에서 주변 단어들을 입력으로, 중심 단어를 타겟으로 사용하여 훈련 데이터를 구성합니다. 이 때, 입력과 타겟은 원-핫 인코딩된 벡터로 표현됩니다.
모델 훈련: 네트워크를 통해 주변 단어의 평균 임베딩을 계산하고, 이를 사용하여 중심 단어를 예측합니다. 학습 과정에서는 손실 함수를 최소화하고 가중치를 업데이트합니다.
각 문장 또는 문서의 길이는 직접적으로 모델의 어휘 사전 크기에 영향을 주지 않습니다. 대신, 훈련 데이터 전체에서 유니크한 단어의 수가 어휘 사전의 크기를 결정합니다. 
따라서 모델을 설계할 때는 전체 훈련 데이터 세트를 고려하여 충분한 크기의 어휘 사전을 준비해야 합니다.`;

// let testWord = '테스트할 한 단어를 입력합니다. 테스트 결과로 입력된 단어의 주변 단어가 리턴 됩니다.';
let testWord = '사전은';
let testDoc = `
    테스트할 문장을 입력하며, (.)으로 구분하여 여러문장 입력합니다.
    테스트 결과로 학습된 데이터들과 유사한 문장만을 리턴합니다. ( 위에서 [유사도 > 0.5] 이상의 임계값 데이터를 리턴 )
`;

wordSimilarity(lrnWord, testWord)
// docSimilarity(lrnWord, testDoc);