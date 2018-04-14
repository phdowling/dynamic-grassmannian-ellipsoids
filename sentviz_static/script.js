console.log("TEST");



function doEmbed(){
    console.log("doEmbed called");
    var sent = document.getElementById('textbox').value;
    httpGetAsync("/embed?sentence=" + sent, embedCallback);
}
function embedCallback(responseText){
    responseJson = JSON.parse(responseText);
    console.log(responseJson);

    var container = document.getElementById('vizControlsArea');
    container.innerHTML = "";
    var iDiv = document.createElement('div');
    iDiv.style.background = "grey";
    iDiv.style.height = "20px";
    iDiv.style.margin = "3px";
    iDiv.style.width = "20px";
    iDiv.setAttribute("onclick", "setTopic(-1)");
    container.appendChild(iDiv);

    var arrayLength = Object.keys(responseJson.sent_topics).length;
    for (var i = 0; i < arrayLength; i++) {
        var currentTopic = responseJson.sent_topics[i];
        var iDiv = document.createElement('div');
        iDiv.setAttribute("id", "topic_" + i);
        iDiv.setAttribute("class", "topic_button");
        iDiv.style.background = "black";
        iDiv.style.height = "20px";
        iDiv.style.margin = "3px";
        iDiv.style.width = currentTopic.importance * 800 + "px";
        iDiv.setAttribute("onclick", "setTopic(" + i + ")");
        container.appendChild(iDiv);
    }

    setTopic(0);

    return responseText;
}

function setTopic(i){
    topic_i = i;
    applyTopic();
}

function applyTopic(){
    var container = document.getElementById('wordVizArea');
    container.innerHTML = "";
    var buttons = document.getElementsByClassName('topic_button');
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].style.background = "black";
        buttons[i].style.border = "2px solid black";
    }

    if(topic_i != -1){
        var button = document.getElementById('topic_' + topic_i);
        button.style.border = "2px solid black";
        button.style.background = "white";
    }

    var found = responseJson.words_found;
    var selectedTopic = responseJson.sent_topics[topic_i];
    var arrayLength = responseJson.words.length;
    for (var i = 0; i < arrayLength; i++) {
        var word = responseJson.words[i];
        var found = responseJson.words_found[i];
        var iSpan = document.createElement('span');

        iSpan.innerHTML = word + " ";
        iSpan.id = 'block_' + i;
        if (found){
            var bestTopicForWord = responseJson.topics_by_word[word];
            iSpan.className = 'found';
            iSpan.setAttribute("onclick", "setTopic(" + bestTopicForWord + ")");
            if (topic_i == -1){
                iSpan.style.fontSize = "150%";
            }else{
                iSpan.style.fontSize = selectedTopic.words[word] * 250 + "%";
            }

        }else{
            iSpan.className = 'notFound';
        }

        container.appendChild(iSpan);
    };
}

function doSimpleSearch(){
    console.log("doSimpleSearch called");
    var words = responseJson.words.join("|");
    httpGetAsync("/query_weighted?words=" + words, simpleSearchCallback);
}

function doTopicSearch(){
    console.log("doTopicSearch called");
    var vec = responseJson.topic_vectors[topic_i].join("|");
    httpGetAsync("/query_by_vec?vec=" + vec, topicSearchCallback);
}

function doWeightedTopicSearch(){
    console.log("doTopicSearch called");
    var words_orig = responseJson.words;
    var weights = [];
    var words = [];
    for (var i = 0; i < words_orig.length; i++){
        var wordWeight = responseJson.sent_topics[topic_i].words[words_orig[i]] || null;
        if (wordWeight != null){
            weights.push(wordWeight);
            words.push(words_orig[i]);
        }

    }

    httpGetAsync("/query_weighted?words=" + words.join("|") + "&weights=" + weights.join("|"), topicSearchCallback);
}

function topicSearchCallback(responseText){
    var searchJson = JSON.parse(responseText);
    var container = document.getElementById("topicSearchContainer");
    container.innerHTML = "";
    for (var i = 0; i< searchJson.query_results.length; i++){
        var res = searchJson.query_results[i];
        var searchResult = document.createElement("div");
        searchResult.setAttribute("class", "searchresult");
        searchResult.innerHTML =  '<a href="' + res.url + '" target="_blank">' + res.similarity + '</a><br/>'+res.sent_text;
        container.appendChild(searchResult);
    }
    console.log("topic search callback");
    console.log(responseText);
}

function simpleSearchCallback(responseText){
    var searchJson = JSON.parse(responseText);
    var container = document.getElementById("simpleSearchContainer");
    container.innerHTML = "";
    for (var i = 0; i< searchJson.query_results.length; i++){
        var res = searchJson.query_results[i];
        var searchResult = document.createElement("div");
        searchResult.setAttribute("class", "searchresult");
        searchResult.innerHTML =  '<a href="' + res.url + '" target="_blank">' + res.similarity + '</a><br/>'+res.sent_text;
        container.appendChild(searchResult);
    }
    console.log("simple search callback");
    console.log(responseText);
}

function httpGetAsync(theUrl, callback)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", theUrl, true); // true for asynchronous
    xmlHttp.send(null);
}