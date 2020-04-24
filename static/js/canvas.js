let canvas = document.getElementById("drawing-board");
let ctx = canvas.getContext("2d");
let eraser = document.getElementById("eraser");
let clear = false;
let brush = document.getElementById("brush");
let resetCanvas = document.getElementById("clear");
let save = document.getElementById("save");
let submit = document.getElementById("submit");

let offsetLeft = canvas.getBoundingClientRect().left;
let offsetTop = canvas.getBoundingClientRect().top;

listenToUser(canvas);

eraser.onclick = function () {
    clear = true;
    this.classList.add("active");
    brush.classList.remove("active");
}

brush.onclick = function () {
    clear = false;
    this.classList.add("active");
    eraser.classList.remove("active");
}

resetCanvas.onclick = function () {
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
}

save.onclick = function () {
    let imgURL = canvas.toDataURL();
    let saveA = document.createElement("a");
    document.body.appendChild(saveA);
    saveA.href = imgURL;
    saveA.download = "zspic" + (new Date).getTime();
    saveA.target = "_blank";
    saveA.click();
}

submit.onclick = function() {
    canvasSendToServer();
}

function drawCircle(x, y, radius) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (clear) {
        ctx.clip();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    }
}

function drawLine(x1, y1, x2, y2) {
    ctx.lineWidth = 28;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (clear) {
        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx_move(x1, y1, x2, y2);
        ctx.clip();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    } else {
        ctx.lineWidth = 14;
        ctx_move(x1, y1, x2, y2);
    }
}

function ctx_move(x1, y1, x2, y2) {
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.closePath();
}

function listenToUser(canvas) {
    let painting = false;
    let lastPoint = { "x": undefined, "y": undefined };

    canvas.onmousedown = function (e) {
        painting = true;
        let x = e.clientX - offsetLeft;
        let y = e.clientY - offsetTop;
        lastPoint = { "x": x, "y": y };
        ctx.save();
        drawCircle(x, y, 0);
        $("#predict").text("");
    }

    canvas.onmouseup = function () {
        painting = false;
    }

    canvas.onmousemove = function (e) {
        if (painting) {
            let x = e.clientX - offsetLeft;
            let y = e.clientY - offsetTop;
            let newPoint = { "x": x, "y": y };
            drawLine(lastPoint.x, lastPoint.y, newPoint.x, newPoint.y, clear);
            lastPoint = newPoint;
        }
    }

    canvas.onmouseup = function () {
        painting = false;
    }

    canvas.mouseleave = function () {
        painting = false;
    }
}

function canvasSendToServer() {
    let imgURL = canvas.toDataURL("image/png");
    let imgDataB64 = imgURL.substring(22);
    var data = { imgB64: imgDataB64 }
    var sendData = JSON.stringify(data);

    $.ajax({
        url: "http://localhost:9000/MNIST",
        type: "POST",
        data: sendData,
        async: true,
        cashe: false,
        contentType: "application/json",
        processData: false,
        beforeSend: function() {
            $("#predict").text("predicting");
        },
        success: function (data) {
            $("#predict").text(data.result);
        },
        error: function (data) {
            alert("fail!");
        }
    })
}