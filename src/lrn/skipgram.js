
const fs = require('fs');


/**
 * 파일 read
 */
const textPath = './src/data/morphemeTarget/';
const dataPath = './src/data/vector/';
const vectorFileName = 'vector.txt';
const indexFileName = 'index.txt';

let isExists = fs.existsSync(dataPath);

if(isExists){
    fs.rmSync(dataPath,  { recursive: true, force: true }, (err) => {
        if (err) {
          console.log(err)
        } else {
          console.log('Dir is deleted.');
        }
    });

    fs.mkdirSync(dataPath);
}

let textFiles = fs.readdirSync(textPath);

if(textFiles.length < 1) return;


/**
 * run time log
 */
var Time = {};
function timeS(tag){
    Time[tag] ? Time[tag].p = process.hrtime() : Time[tag] = {p : process.hrtime()};
}

function timeE(tag) {
    const diff = process.hrtime(Time[tag].p);
    const sTime = (diff[0] + diff[1] / 1e9).toFixed(3);

    // 성능 저하로 인한 임시 주석
    // const msTime = (diff[0] * 1000 + diff[1] / 1e6).toFixed(3);
    // Time[tag].p = undefined;
    // Time[tag].t = msTime;
    // Time[tag].s = Time[tag].s > 0 ? +Time[tag].s + +msTime : msTime;

    return sTime;
}

function timeA(tag, cnt){
    if(!Time[tag]?.s) return;
    const avg = Time[tag].s / cnt;
    Time[tag] = {a : avg};
    return console.log(`runTime ::: ${tag} :: ${Time[tag].a}`);
}


/**
 * 메인
 */
(function(){

    const epoch = 300;
    const windowSize = 5;
    const dim = 50;
    const learningRate = 0.001;
    const numSamples = 5;
    
    var splitAllWord = [];  // 전체 파일
    var wordWeights = [];   // 단어 벡터(임베딩)
    var vocabulary = [];    // 단어 중복제거 목록
    var frequentWords = {}; // 단어 출현 빈도수
    var wordToIndex = {};   // 단어 인덱스 (기존 indexOf 퍼포먼스 저조)

    // 파일 읽기
    textFiles.forEach((file, index) => {
        let f = fs.readFileSync(textPath + '/' + file);

        // let words = f.toString().split(/[ \t\r\n]/g);
        let words = JSON.parse(f.toString());

        splitAllWord.push(words);
        vocabulary = Array.from(new Set([...vocabulary, ...words]));
        words.forEach((word, index) => { frequentWords[word] = {cnt : (frequentWords[word]?.cnt || 0) + 1 }});

        // 빈도수에 따른 학습확률
        let frqTotal = Object.keys(frequentWords).length;
        Object.keys(frequentWords).forEach((word, index) => { 
            frequentWords[word]['probability'] = 1 - Math.sqrt( 1e-3 / ( frequentWords[word].cnt / frqTotal ) );
        });
    });
    
    vocabulary.forEach((word, index) => { wordToIndex[word] = index });
    wordWeights = initWord(vocabulary);
    
    console.log(`전체 단어수 :: ${vocabulary.length}`);

    // 학습수
    for(let i=0; i<epoch; i++){
        
        // 전체 손실
        let totalLoss = 0;
        let totalPairs = 0;
        let timeAvgCnt = 0;
        timeS('RunTime');

        // 파일목록
        for(words of splitAllWord){

            // 빈도수 학습 확률
            let frequentRand =  Math.random();

            // 파일의 단어 목록
            for(let wordIndex = 0; wordIndex < words.length; wordIndex++){
                
                // 중심단어
                let cntrWord = words[wordIndex];
                let cntrVec = wordWeights[wordToIndex[cntrWord]];

                // 랜덤한 부정단어 목록 ( 긍정단어외 )
                let negatives = getNegativeSamples(wordIndex, words);
                
                // 중심단어의 주변단어 목록 ( windowSize *2 )
                for(let contextIndex = 0; contextIndex < windowSize*2; contextIndex++){
                    
                    // 중심기준 긍정단어
                    timeS('pstvIndex');
                    let pstvIndex = wordIndex - windowSize + (windowSize <= contextIndex ? contextIndex + 1 : contextIndex);
                    let pstvWord = words[pstvIndex];
                    timeE('pstvIndex');

                    // 빈도수 높은 단어 정교한 예측을 위해 일정확률로 스킵
                    let isDiscardProbability = frequentRand < frequentWords[cntrWord].probability;
                    if(isDiscardProbability || wordIndex == pstvIndex || pstvIndex < 0 || !pstvWord) continue;
                    
                    // 긍정단어 학습
                    timeS('pstvUpdate');
                    let pstcVec = wordWeights[wordToIndex[pstvWord]];
                    let positiveScore = dotProduct(cntrVec, pstcVec);
                    let positiveGrad = sigmoid(positiveScore) - 1;
                    updateWeights(cntrVec, pstcVec, positiveGrad);
                    timeE('pstvUpdate');
                    
                    // 부정 단어 학습
                    timeS('ngtvUpdate');
                    let ngtvVecs = [];
                    negatives.forEach(negativeIndex => {
                        let ngtvVec = wordWeights[wordToIndex[words[negativeIndex]]];
                        let negativeScore = dotProduct(cntrVec, ngtvVec);
                        let negativeGrad = sigmoid(negativeScore) - 0;
                        updateWeights(cntrVec, ngtvVec, negativeGrad);
                        ngtvVecs.push(ngtvVec);
                        totalPairs++;
                    });
                    timeE('ngtvUpdate');
    
                    // 손실률 ( 성능저하 마지막 카운트에만 측정 )
                    if(i+1 == epoch){
                        timeS('skipGramLoss');
                        totalLoss += skipGramLoss(cntrVec, pstcVec, ngtvVecs);
                        timeE('skipGramLoss');
                    }
                    
                    totalPairs++;
                    timeAvgCnt++;
                    
                }
            }
        }

        timeA('pstvIndex', timeAvgCnt);
        timeA('pstvUpdate', timeAvgCnt);
        timeA('ngtvUpdate', timeAvgCnt);
        timeA('skipGramLoss', timeAvgCnt);
        console.log(`
            :: Total Count ${i+1} / ${epoch}
            :: One Cycle Time ${timeE('RunTime')}s 
            :: Loss Total ${totalLoss / totalPairs}
        `);
    }

    // 저장
    fs.writeFileSync(dataPath + vectorFileName, JSON.stringify(wordWeights));
    fs.writeFileSync(dataPath + indexFileName, JSON.stringify(vocabulary));



    /**
     * ===========================================================================
     *  함수
     */
    
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
    function skipGramLoss(centerVec, positiveVec, negativeVecs) {
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

    // 임베딩 랜덤(0 ~ 0.5) 초기화
    function initWord(words){
        let embeddings = new Array(words.length);
        for (let i = 0; i < words.length; i++) {
            embeddings[i] = new Array(dim).fill().map(() => Math.random() - 0.5);
        }
        return embeddings;
    }

})();


