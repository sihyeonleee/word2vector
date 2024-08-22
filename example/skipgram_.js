/**
 * skip-gram 의 동작 방식이나 기본적인 이해를위해 개인 학습의 목적으로 만든 프로그램입니다.
 * 구현은 https://reniew.github.io/22/ 외 여러 사이트 참고하여 만들었습니다.
 * 
 * ## 기본원리
 * - skip-gram 은 CBOW 와 다르게 중심단어를 기준으로 window size 만큼 주변단어들과 한쌍을 이루어 긍정단어를 학습하며, window size 를 벗어나는 단어를 무작위 선택(1개이상)하여 부정단어로 학습시킵니다.
 * - 추가로 the, is, a 와 같은 자주 등장하는 단어에 대해서 **Subsampling Frequent words**를 사용해 일정 확률로 학습을 제외 합니다. ( 확률성 제외 이기때문에 학습을 안시키는게 아님 )
 * - 학습은 시그모이드(sigmoid)를 사용하며, 손실률은 크로스엔트로피(crossEntropy)로 측정, 테스트는 코사인유사도(cosineSimilarity)를 사용했습니다.
 * 
 * ## 메인소스
 * - 메인함수인 lrnStart() 함수는 3중 반복문으로 동작하며 학습수인 epoch, 테스트문장의 각 중심단어, 각 중심단어와 쌍을이룰 주변단어(window size * 2) 순으로 순회합니다.
 * - 원핫벡터의 입력값과 Hidden Layer 의 행렬곱은 결국 해당 단어의 벡터값을 가져오기 때문에 v * n = W 과정은 생략했습니다.
 * - 해당소스는 학습목적의 소스이며 데이터 정제, 문장 토큰화, 문장간 학습, 성능 튜닝 등을 고려하지 않았습니다. 이해의 목적으로 봐주시길 바랍니다.
 * 
 * ## 소스코드
 * - 자세한 설명은 주석으로 대체 합니다.
 */

const epoch = 100;			// 학습수
const dim = 50;				// 차원수
const windowSize = 5;		// 주변단어 크기(단어기준 앞, 뒤로 가져오기 때문에 X2 하여 사용)
const learningRate = 0.01;	// 학습률로 0.01은 테스트용이며 실제론 더 낮은값을 사용합니다.
const numSamples = 5;		// 부정단어를 추출할 단어의 수 입니다.

var wordWeights = [];   // 단어의 벡터( 이 배열을 업데이트 하며 학습시킵니다. )
var vocabulary = [];    // 중복 제거된 단어의 인덱스
var frequentWords = {}; // 단어 출현의 빈도수

function lrnStart(str){

    let words = str.split(' '); // 문장분리 ( 단순 띄어쓰기 분리 )
    vocabulary = Array.from(new Set(words)); // 중복제거된 단어들의 인덱스
    wordWeights = initWord(vocabulary); // 임베딩 초기화 ( 0~0.5 사이의 낮은 값으로 dim 수만큼 초기화 합니다. )
    words.forEach((word, index) => { frequentWords[word] = {cnt : (frequentWords[word]?.cnt || 0) + 1 }}); // 단어의 빈도수 카운트
    
    for(let i=0; i<epoch; i++){

        // 손실률 측정을 위한 변수
        let totalLoss = 0;
        let totalPairs = 0;

        for(let wordIndex = 0; wordIndex < words.length; wordIndex++){
            for(let contextIndex = 0; contextIndex < windowSize*2; contextIndex++){

                let cntrWord = words[wordIndex];
				
                // 중심단어를 기준으로 좌측부터 긍정단어(window 크기만큼 인접한 단어) 가져오기
                let pstvIndex = wordIndex - windowSize + (windowSize <= contextIndex ? contextIndex + 1 : contextIndex);
                let pstvWord = words[pstvIndex];

                // 빈번한 단어의 등장은 정교한 예측을 위해 일정확률로 스킵 ( Subsampling Frequent words ) 
                let isDiscardProbability = frequentWordToPer(frequentWords, cntrWord);
                if(isDiscardProbability || wordIndex == pstvIndex || pstvIndex < 0 || !pstvWord) continue;

                // 중심단어와 긍정단어의 벡터
                let cntrVec = wordWeights[vocabulary.indexOf(cntrWord)];
                let pstcVec = wordWeights[vocabulary.indexOf(pstvWord)];
                
                // 랜덤한 부정단어 목록 ( 긍정단어외 numSamples의 수만큼 선택 )
                let negatives = getNegativeSamples(wordIndex, words);
                let ngtvVecs = [];

                // 긍정 단어 학습
                let positiveScore = dotProduct(cntrVec, pstcVec); // 두벡터간의 내적
                let positiveGrad = sigmoid(positiveScore) - 1;
                updateWeights(cntrVec, pstcVec, positiveGrad);

                // 부정 단어 학습
                negatives.forEach(negativeIndex => {
                    let ngtvVec = wordWeights[vocabulary.indexOf(words[negativeIndex])];
                    let negativeScore = dotProduct(cntrVec, ngtvVec);
                    let negativeGrad = sigmoid(negativeScore);
                    updateWeights(cntrVec, ngtvVec, negativeGrad);
					
					// 손실률확인을 위함
                    ngtvVecs.push(ngtvVec);
                    totalPairs++;
                });

                // 손실률 확인
                let loss = crossEntropy(cntrVec, pstcVec, ngtvVecs);
                totalLoss += loss;
                totalPairs++;
            }
        }
		// 각 epoch 마다 손실률을 측정합니다. ( 0에 가까울수록 좋은 성능을 보인다고 합니다. )
        loss = totalLoss / totalPairs;
        console.log(`손실률 ${loss} ::: ${i+1} / ${epoch}`)
    }
}

// 가중치 업데이트
function updateWeights(centerVec, contextVec, grad){
    for (let i = 0; i < dim; i++) {
        centerVec[i] -= learningRate * contextVec[i] * grad;
        contextVec[i] -= learningRate * centerVec[i] * grad;
    }
}

// 부정 단어 인덱스 리스트
function getNegativeSamples(wordIndex, words) {
    const negativeSamples = [];

    // 중복 없이 부정 샘플 선택
    while (negativeSamples.length < numSamples) {
        const randomIndex = Math.floor(Math.random() * words.length);
        // 부정단어 선정기준 : 본인X, 윈도우 사이즈 내 X
        if (!negativeSamples.includes(randomIndex) && (randomIndex < wordIndex - windowSize || wordIndex + windowSize < randomIndex)) {
            negativeSamples.push(randomIndex);
        }
    }
    return negativeSamples;
}

// 시그모이드 함수 정의
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// 두 벡터의 내적 계산
function dotProduct(vec1, vec2) {
    return vec1.reduce((acc, curr, idx) => acc + curr * vec2[idx], 0);
}

// 손실률 계산
function crossEntropy(centerVec, positiveVec, negativeVecs) {
    let loss = 0;

    // 긍정적인 예측에 대한 손실 계산
    const posScore = dotProduct(centerVec, positiveVec);
    const posLoss = -Math.log(sigmoid(posScore));
    loss += posLoss;

    // 부정적인 예측에 대한 손실 계산 
    negativeVecs.forEach(negativeVec => {
        const negScore = dotProduct(centerVec, negativeVec);
        const negLoss = -Math.log(1 - sigmoid(negScore));
        loss += negLoss;
    });

    return loss;
}

// Subsampling Frequent words
// 빈번하게 등장하는 단어의 학습확률계산
function frequentWordToPer(words, word){
    words[word].probability = words[word].probability || 1 - Math.sqrt(1e-3/(words[word].cnt/Object.keys(words).length));
    return Math.random() < words[word].probability;
}

// 임베딩 랜덤(0 ~ 0.5) 초기화
function initWord(words){
    let embeddings = new Array(words.length);
    for (let i = 0; i < words.length; i++) {
        embeddings[i] = new Array(dim).fill().map(() => Math.random() - 0.5);
    }
    return embeddings;
}

// ====================================== 테스트 ======================================================

// 코사인 유사도
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

function wordSimilarity(str, testWord){

    lrnStart(str);

    let similarityList = [];
    let similarity = -Infinity;
    let similarIndex = 0;
    let similarWord = '';
    let trgetIndex = vocabulary.indexOf(testWord);
	
    if(trgetIndex < 1) {
        console.log('매칭불가 :: 신규 단어 재학습 필요')
        return '매칭불가 :: 신규 단어 재학습 필요';
    }
	
    console.log(`테스트 단어! ::: ${vocabulary[trgetIndex]} / 위치 ::: ${trgetIndex}`);
	
    for (var w=0; w<vocabulary.length; w++) {
        if (w !== trgetIndex) {
            similarity = cosineSimilarity(wordWeights[w], wordWeights[trgetIndex]);
            similarWord = vocabulary[w];
            similarIndex = w;
            similarityList.push({
                similarity : similarity,
                similarWord : similarWord,
                similarIndex : similarIndex,
                msg : `유사 단어 : ${similarWord} / 유사 단어 위치 : ${similarIndex} / 유사도 : ${similarity}`
            });
        }
    }

    similarityList.sort((a,b) => b.similarity - a.similarity);

    console.log(JSON.stringify({
		result : similarityList.slice(0, 10),
		input : `테스트 단어 ::: ${vocabulary[trgetIndex]} / 위치 ::: ${trgetIndex}`
    }, null, 2));
}

// ====================================== 실행 ======================================================

// 학습데이터 ( 임시 )
let lrnWord = `Here the army has its ranges where it tests missiles and other weaponry.
The drug must first be tested in clinical trials to see if it works on other cancers.
They are not really testing pupils; they are testing the teachers.
She decided to test herself with a training run in London.
Out of a total of 2,602 pupils, only 922 passed the test.
She had sold her bike, taken a driving test and bought a car.
She may be testing her mother to see how much she can take before she throws her out.
It is a commonplace fact that holidays are a major test of any relationship.`;


/**
 * 테스트할 한 단어를 입력합니다. 테스트 결과로 입력된 단어의 주변 단어가 리턴 됩니다.
 */
let testWord = 'driving';

// 테스트 시작
wordSimilarity(lrnWord, testWord);