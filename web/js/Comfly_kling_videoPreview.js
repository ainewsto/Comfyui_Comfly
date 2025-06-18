import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object");
        return;
    }
    if (property in object) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r;
        };
    } else {
        object[property] = callback;
    }
}

function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
        let optNew = [];
        try {
            const previewWidget = this.widgets.find((w) => w.name === "videopreview");

            let url = null;
            if (previewWidget?.videoEl?.hidden === false && previewWidget.videoEl.src) {
                url = previewWidget.videoEl.src;
            }
            if (url) {
                optNew.push(
                    {
                        content: "Open preview",
                        callback: () => {
                            window.open(url, "_blank");
                        },
                    },
                    {
                        content: "Save preview",
                        callback: () => {
                            const a = document.createElement("a");
                            a.href = url;
                            a.setAttribute("download", new URLSearchParams(previewWidget.value.params).get("filename"));
                            document.body.append(a);
                            a.click();
                            requestAnimationFrame(() => a.remove());
                        },
                    }
                );
            }
            if (options.length > 0 && options[0] != null && optNew.length > 0) {
                optNew.push(null);
            }
            options.unshift(...optNew);
        } catch (error) {
            console.log(error);
        }
    });
}

function previewVideo(node, file, type) {
    const params = {
        filename: file,
        type: type,
        force_size: "256x?"
    };

    const videoUrl = api.apiURL("/view?" + new URLSearchParams(params));

    // 🟢 若已有 video preview widget，复用它
    const existing = node.widgets?.find(w => w.name === "videopreview");
    if (existing && existing.videoEl) {
        existing.videoEl.src = videoUrl;
        existing.videoEl.hidden = false;
        existing.parentEl.hidden = false;
        existing.value.params = params;
        fitHeight(node);
        return;
    }

    // 🆕 创建 widget（首次或缺失时）
    const element = document.createElement("div");
    const previewWidget = node.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(v) {
            element.value = v;
        },
    });

    previewWidget.computeSize = function (width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
            let height = (node.size[0] - 20) / this.aspectRatio + 10;
            if (!(height > 0)) height = 0;
            this.computedHeight = height + 10;
            return [width, height];
        }
        return [width, -4];
    };

    previewWidget.value = { hidden: false, paused: false, params };
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "video_preview";
    previewWidget.parentEl.style.width = "100%";
    element.appendChild(previewWidget.parentEl);

    previewWidget.videoEl = document.createElement("video");
    previewWidget.videoEl.controls = true;
    previewWidget.videoEl.loop = false;
    previewWidget.videoEl.muted = false;
    previewWidget.videoEl.style.width = "100%";
    previewWidget.videoEl.src = videoUrl;

    previewWidget.videoEl.addEventListener("loadedmetadata", () => {
        previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
        fitHeight(node);
    });

    previewWidget.videoEl.addEventListener("error", () => {
        previewWidget.parentEl.hidden = true;
        fitHeight(node);
    });

    previewWidget.videoEl.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden;
    previewWidget.videoEl.hidden = false;

    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    previewWidget.parentEl.appendChild(previewWidget.videoEl);
}

app.registerExtension({
    name: "Comfly.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name === "Comfly_kling_videoPreview") {
            nodeType.prototype.onExecuted = function (data) {
                previewVideo(this, data.video[0], data.video[1]);
            };
            addPreviewOptions(nodeType);
        }
    }
});
