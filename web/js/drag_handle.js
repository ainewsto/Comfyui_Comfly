export const dragHandle={createButton(e,t){const n=document.createElement("div");n.style.position="relative",n.style.cursor="move";const s=document.createElement("button");s.style.backgroundImage="url('https://api.comfly.chat/wp-content/uploads/2024/07/Comfly-t拷贝.png')",s.style.backgroundSize="cover",s.style.backgroundRepeat="no-repeat",s.style.backgroundPosition="center",s.textContent="",s.style.width="38px",s.style.height="38px",s.style.padding="10px",s.style.borderRadius="50%",s.style.margin="5px",s.style.backgroundColor="transparent",s.style.border="none",s.style.boxShadow="0 0 10px 8px rgba(255, 253, 1, 0.7)",s.style.transition="transform 0.2s, box-shadow 0.2s",s.style.animation="flame 1.5s infinite alternate";const o=document.createElement("button");return o.textContent="×",o.style.position="absolute",o.style.top="-5px",o.style.right="-5px",o.style.backgroundColor="rgba(200, 200, 200, 0.7)",o.style.color="black",o.style.border="none",o.style.borderRadius="50%",o.style.width="20px",o.style.height="20px",o.style.fontSize="16px",o.style.cursor="pointer",o.style.display="none",o.style.zIndex="1001",n.appendChild(s),n.appendChild(o),n.addEventListener("mouseover",(()=>{o.style.display="block",s.style.transition="transform 0.5s ease",s.style.transform="rotate(360deg)"})),n.addEventListener("mouseout",(e=>{n.contains(e.relatedTarget)||(o.style.display="none",s.style.transform="rotate(0deg)")})),n.addEventListener("mousedown",(()=>{s.style.transform="scale(0.95)",s.style.boxShadow="0 0 6px 5px rgba(255, 253, 1, 0.7)"})),n.addEventListener("mouseup",(()=>{s.style.transform="scale(1)",s.style.boxShadow="0 0 12px 8px rgba(255, 253, 1, 0.7)"})),n.addEventListener("click",(e=>{e.target!==o&&("none"===t.style.display?(t.style.display="flex",t.style.opacity=0,t.style.transition="opacity 0.5s ease-in-out",setTimeout((()=>{t.style.opacity=1}),10)):(t.style.opacity=0,setTimeout((()=>{t.style.display="none"}),500)))})),o.addEventListener("click",(t=>{t.stopPropagation(),this.hideContainer(e)})),this.makeDraggable(e,n),"true"===localStorage.getItem("comflyContainerHidden")&&(e.style.display="none"),n},hideContainer(e){e.style.display="none",localStorage.setItem("comflyContainerHidden","true")},showContainer(){const e=document.querySelector(".comfly-container");e&&(e.style.display="flex",localStorage.setItem("comflyContainerHidden","false"))},makeDraggable(e,t){let n,s,o,a,l=!1,r=0,d=0;const i=e=>{"touchstart"===e.type?(o=e.touches[0].clientX-r,a=e.touches[0].clientY-d):(o=e.clientX-r,a=e.clientY-d),(e.target===t||t.contains(e.target))&&(l=!0)},y=e=>{o=n,a=s,l=!1},c=t=>{l&&(t.preventDefault(),"touchmove"===t.type?(n=t.touches[0].clientX-o,s=t.touches[0].clientY-a):(n=t.clientX-o,s=t.clientY-a),r=n,d=s,p(n,s,e))},p=(e,t,n)=>{n.style.transform=`translate3d(${e}px, ${t}px, 0)`};let u;const m=(e,t)=>{u||(u=!0,setTimeout((()=>{e(),u=!1}),t))};t.addEventListener("mousedown",i),t.addEventListener("touchstart",i),document.addEventListener("mousemove",(e=>{m((()=>c(e)),10)})),document.addEventListener("touchmove",(e=>{m((()=>c(e)),10)})),document.addEventListener("mouseup",y),document.addEventListener("touchend",y)}};const style=document.createElement("style");style.textContent="\n@keyframes flame {\n    0% {\n        box-shadow: 0 0 8px 3px rgba(255, 253, 1, 0.5);\n    }\n    100% {\n        box-shadow: 0 0 8px 5px rgba(255, 165, 0, 0.7);\n    }\n}\n",document.head.append(style);let isAPressed=!1;document.addEventListener("keydown",(e=>{"a"===e.key.toLowerCase()&&(isAPressed=!0),isAPressed&&"s"===e.key.toLowerCase()&&(e.preventDefault(),dragHandle.showContainer())})),document.addEventListener("keyup",(e=>{"a"===e.key.toLowerCase()&&(isAPressed=!1)}));
