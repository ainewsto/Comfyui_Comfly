import{app}from"../../../scripts/app.js";import{helpButton}from"./help_button.js";import{dragHandle}from"./drag_handle.js";import{manageDependenciesButton}from"./Comfly_manager.js";import{chatButton}from"./chat_button.js";const ComflyExtension={name:"Comfly",async init(){console.log("[Comfly]","Extension initialized"),await chatButton.init(),this.createUI()},createUI(){const t=document.createElement("div");t.className="comfly-container",t.style.position="fixed",t.style.left="95px",t.style.bottom="20px",t.style.zIndex="1000",t.style.display="flex",t.style.flexDirection="row",t.style.alignItems="center",document.body.appendChild(t);const e=document.createElement("div");e.style.display="none",e.style.flexDirection="row",e.style.alignItems="center",e.style.marginLeft="10px";const n=dragHandle.createButton(t,e);t.appendChild(n);const o=chatButton.createButton(t);e.appendChild(o);const a=manageDependenciesButton.createButton();e.appendChild(a);const l=helpButton.createButton();e.appendChild(l),t.appendChild(e),"true"===localStorage.getItem("comflyContainerHidden")&&(t.style.display="none")}};app.registerExtension(ComflyExtension);
