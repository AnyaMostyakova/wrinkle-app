const fileInput = document.getElementById("fileInput")
const analyzeBtn = document.getElementById("analyzeBtn")
const previewBox = document.getElementById("previewBox")
const previewImage = document.getElementById("previewImage")
const uploadText = document.getElementById("uploadText")

// -----------------------------------
// Выбор файла
// -----------------------------------

fileInput.addEventListener("change", function(){

    if(this.files.length > 0){

        let file = this.files[0]

        uploadText.innerText = file.name

        analyzeBtn.disabled = false

        let reader = new FileReader()

        reader.onload = function(e){

            previewImage.src = e.target.result
            previewBox.classList.remove("hidden")

        }

        reader.readAsDataURL(file)

    }

})


// -----------------------------------
// Анализ изображения
// -----------------------------------

analyzeBtn.addEventListener("click", async function(){

    let file = fileInput.files[0]

    if(!file){
        alert("Выберите изображение")
        return
    }

    let formData = new FormData()

    formData.append("file", file)

    document.getElementById("loading").classList.remove("hidden")

    let response = await fetch("/analyze",{

        method:"POST",
        body:formData

    })

    let data = await response.json()

    document.getElementById("loading").classList.add("hidden")

    document.getElementById("result").classList.remove("hidden")

    document.getElementById("originalImage").src = data.original
    document.getElementById("resultImage").src = data.result

    let percent = data.wrinkle_percent.toFixed(2)

    document.getElementById("wrinklePercent").innerText =
        "Процент морщин: " + percent + "%"

    document.getElementById("wrinkleBar").style.width =
        percent + "%"

})


// -----------------------------------
// Новый анализ
// -----------------------------------

function resetAnalysis(){

    document.getElementById("result").classList.add("hidden")
    document.getElementById("previewBox").classList.add("hidden")

    fileInput.value = ""

    analyzeBtn.disabled = true

    uploadText.innerText = "Выберите фото"

    document.getElementById("wrinkleBar").style.width = "0%"

}


// -----------------------------------
// Сохранить результат
// -----------------------------------

function downloadResult(){

    let img = document.getElementById("resultImage").src

    let link = document.createElement("a")

    link.href = img
    link.download = "wrinkle_analysis_result.png"

    document.body.appendChild(link)

    link.click()

    document.body.removeChild(link)

}