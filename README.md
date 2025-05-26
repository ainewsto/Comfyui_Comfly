<a name="readme-top"></a>
![资源 1小显示器-svg](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/e36d75e0-2cba-4026-936e-1ba8aba9cc7b)

<div align="center">

<a href="https://comfly.chat"> <img alt="Static Badge" src="https://img.shields.io/badge/Comfyui_forum-online-fffd01.svg"> </a>
# 👋🏻 Welcome to Comfly

</div>



我喜欢comfyui，它就像风一样的自由，所以我取名为：comfly
同样我也喜欢绘画和设计，所以我非常佩服每一位画家，艺术家，在ai的时代，我希望自己能接收ai知识的同时，也要记住尊重关于每个画师的版权问题。
我一直认为ai最好的方式应该是在购买版权的前提下，有序的完成ai的良性发展. 在学习comfyui的这段旅程中，我遇到了，几位可爱的小伙伴。

并且他们格外的温暖，调皮，傻不拉几。虽然他们没有参与这个项目，但是有几位朋友帮忙做测试。
第一张图片由"走走走"修改后的版本.

![新建项目](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/de5d1b7c-f909-4a3d-892e-6f38c56e4e85)


> **Warning**
> 
> 插件只在windows11上测试，mac电脑后续我也会测试，如果其他系统有任何问题，可以提交issues
> 目前界面没有做太多的美化，有时间再说，必要性不到，主要是功能性插件

# 更新 Update：

20250518：

`Jimeng即梦视频节点`: 新增ComflyJimengVideoApi节点。即梦3.0视频，按次收费，5s是0.6元，10s是1.2元。


20250518：

`Kling节点`: 可灵节点新增kling-v2-master的可灵2.0模型。价格很贵，按需使用。

20250429：

`Chatgpt节点`: Comfly_gpt_image_1_edit新增chats输出口，输出多轮对话。
新增clear_chats,当为Ture的时候，只能image输入什么图片修改什么图片，不支持显示上下文对话。
当为Flase的时候，支持对上一次生成的图片进行二次修改。支持显示上下文对话。并且支持多图模式下新增图片参考。

<details>
<summary>查看更新/Update </summary>  
 
![2eaf76b077612170647f6861e43e2af](https://github.com/user-attachments/assets/1c4c484f-c3c6-48c6-96c5-58c4ef4e59d5)

![6a43cb051fece84815ac6036bee3a4c](https://github.com/user-attachments/assets/f0fbf71e-8cfb-448e-87cd-1e147bb2f552)

</details> 

20250425：

视频教程： https://www.bilibili.com/video/BV1jxLUz9ECX

`Chatgpt节点`: 
新增Comfly_gpt_image_1和Comfly_gpt_image_1_edit官方gpt_image_1模型api接口节点。

![image](https://github.com/user-attachments/assets/9d08d5fc-dde9-4523-955c-31652a74f1a5)

模型名都是gpt_image_1，区别只是分组不同：

一共四个分组：default默认分组为官方逆向，价格便宜，缺点就是不稳定，速度慢。按次收费。不支持额外参数选择。这个分组的apikey只能用于ComflyChatGPTApi节点。

其他三个组都是官方api组，最优惠的目前是ssvip组。分组需要再令牌里面去修改选择。这3个官方分组优点就是速度快，稳定性高。支持官方参数调整。
缺点就是贵，但是也比官方便宜。大家可以按照自己的情况选择。这3个分组的令牌的apikey只能用在下面2个新节点上面！！！

1. Comfly_gpt_image_1 节点：文生图，有耕读参数调整，支持调整生图限制为low。

2. Comfly_gpt_image_1_edit 节点：图生图，支持mask遮罩，支持多图参考。

<details>
<summary>查看更新/Update </summary>  
 
![3bc790641c44e373aca97ea4a1de47e](https://github.com/user-attachments/assets/1a7a0615-46e5-46b3-af04-32246a23d6f4)

![5efe58fcf7055d675962f40c1ad1cbb](https://github.com/user-attachments/assets/8a90eab5-4242-43bb-ae01-74493b90b6ce)

</details> 

20250424：
`Chatgpt节点`: ComflyChatGPTApi节点新增官方gpt-image-1，按次计费 0.06，
旧版的gpt4o-image，gpt4o-image-vip，sora_image, sora_image-vip可以做为备选。首选gpt-image-1。

`jimeng即梦节点`: 即梦的ComflyJimengApi节点新增参考图生成图片，image url图片链接参考生成图片。
注意：参考图生成图片会额外消耗上传图片的token费用（具体根据你图片大小来，大部分都是0.000几到0.00几元不等。图片链接有时效性，不做长期储存），
这个只适用于你没有image url图片链接的前提下使用。
如果你有image url图片链接，就直接填写在image url里面既可以。

<details>
<summary>查看更新/Update </summary>  
 
![e1abc11e855680b70985ec9f339a967](https://github.com/user-attachments/assets/6d77c103-d35a-4c6b-804a-4b5add172bcf)

![307e5ea0d789b785fd0a60f01f2b8cf](https://github.com/user-attachments/assets/5c8a7984-ae5e-4cbf-aa47-b09bc7e6f8d6)

</details> 

20250422：
`Chatgpt节点`: ComflyChatGPTApi节点新增chats输出口，输出多轮对话。
新增clear_chats,当为Ture的时候，只能image输入什么图片修改什么图片，不支持显示上下文对话。
当为Flase的时候，支持对上一次生成的图片进行二次修改。支持显示上下文对话。

<details>
<summary>查看更新/Update </summary>  

![cad243f2bf4a3aa11163f1a007db469](https://github.com/user-attachments/assets/ef0f6a34-3de7-42a2-8543-c1930575e1bb)

![bd6493050affdf156143c8dc5286988](https://github.com/user-attachments/assets/0906caf3-35ec-4061-bfc9-5f611a19abf2)

![e5b3d375b700dcbf921b12a8aa527c4](https://github.com/user-attachments/assets/75537100-e5d2-403c-b2e0-1f662680092f)


</details> 

20250418：
`jimeng即梦节点`: 新增即梦的ComflyJimengApi节点。
目前只支持文生图，使用的是 https://ai.comfly.chat 的 api key

参数说明：
use_pre_llm：开启文本扩写，会针对输入prompt进行扩写优化，如果输入prompt较短建议开启，如果输入prompt较长建议关闭。
scale：影响文本描述的程度，默认值：2.5，取值范围：[1, 10]
width，height：生成图像的宽和高，默认值：1328，取值范围：[512, 2048]
add_logo：是否添加水印。True为添加，False不添加。默认不添加
opacity：水印的不透明度，取值范围0-1，1表示完全不透明，默认0.3

<details>
<summary>查看更新/Update </summary>  
 
![3c1a498bea1853be7aafda2d7ea41b1](https://github.com/user-attachments/assets/13b84330-25d0-420b-9111-8e653f3ada99)

![16be3f66454ae4a74c7c5bc723f847f](https://github.com/user-attachments/assets/96351739-bbcb-476b-a516-1a6a265151db)

</details> 

`ai模块`: 简单化ai模块，支持上传文件解析（复杂的图片pdf会根据具体模型来确定）.优化交互体验。
设置里面可以设置api key。

<details>
<summary>查看更新/Update </summary>  

![4490ee27d100d5414028f7d0e293748](https://github.com/user-attachments/assets/ae088d19-2289-424d-b030-39411cdfca2a)
![9ae58c50da8676e6286e2018e103083](https://github.com/user-attachments/assets/8f45b580-1981-4f22-924c-3c492b54eb68)
![621183f77a72d4ffaf89abde650f4f6](https://github.com/user-attachments/assets/6d838ad9-3679-401c-86c3-a139f57ad3ee)

</details> 

20250401：
`所有节点`: 所有调用apikey的节点新增apikey输入框。优化可灵图生图视频节点。

20250329：
`Chatgpt节点`: 新增openai的ComflyChatGPTApi节点，。
目前单图和多图输入，文本输入，生成图片，图片编辑.使用的是 https://ai.comfly.chat 的 api key
固定一次生成消耗0.06元（显示是逆向api，稳定性还不高，想尝鲜的可以注册网站用免费送的0.2美金玩玩）
速度不快，因为官网速度也不快，所以需要点耐心。 files输入接口还没有完善，先忽略。
用sora_image现在先对稳定点

<details>
<summary>查看更新/Update </summary>  
 
![fdedd73cffa278d2a8cf81478b58e90](https://github.com/user-attachments/assets/36e78cdd-33b2-41ed-a15c-ad9c1886bede)


![0a2394c0b41efe190a5d0880f4c584b](https://github.com/user-attachments/assets/267fbe73-7113-4120-a829-a7aa2247bd4d)

</details> 


20250325：

`Kling节点`: 新增可灵Comfly_lip_sync对口型节点，生成效果还行吧。速度也一般般。支持中文和英文。

`Gemmi节点`: ComflyGeminiAPI节点resolution新增：object_image size,subject_image size,scene_image size根据输入的图片的尺寸来确定输出图片的尺寸。增加image url输出接口。

`Doubao豆包节点`: ComflySeededit节点文字驱动生成图片，编辑图片。支持添加自己的水印logo。目前只支持单图修改和参考。使用的是 https://ai.comfly.chat 的 api key

用于编辑图像的提示词 。建议：

添加/删除实体：添加/删除xxx（删除图上的女孩/添加一道彩虹）

修改实体：把xxx改成xxx（把手里的鸡腿变成汉堡）

修改风格：改成xxx风格（改成漫画风格）

修改色彩：把xxx改成xx颜色（把衣服改成粉色的）

修改动作：修改表情动作（让他哭/笑/生气）

修改环境背景：背景换成xxx，在xxx（背景换成海边/在星空下）

1：图片格式：JPG(JPEG), PNG, BMP 等常见格式, 建议使用JPG格式.

2：图片要求：小于4.7 MB，小于4096*4096

3：长边与短边比例在3以内，超出此比例或比例相对极端，会导致报错

<details>

<summary>查看更新/Update </summary>

![95836fbdda83551ca81ebc3db93b2d5](https://github.com/user-attachments/assets/0ce70dd3-eb0a-4e2e-bf6c-68642a48288d)

![563e17009b9100533f169aa1d87b37f](https://github.com/user-attachments/assets/48bd20e9-9d87-43d7-aa81-370c7c7f1bec)


</details>

20250321：`Gemmi节点`: 谷歌ComflyGeminiAPI节点支持生成文生多图（最多4张，控制时间）。
支持多图片参考，我是借用google labs的whisk思路，我感觉比较实用，并不需要太多参考图，3种足够.无需谷歌账户和梯子魔法就能用。使用的是 https://ai.comfly.chat 的 api key
<details>
<summary>查看更新/Update </summary>  
 
![微信图片_20250321225149](https://github.com/user-attachments/assets/593f479b-51d5-476e-bcf7-f36c4f01eb29)

</details> 

20250319：`Gemmi节点`: 新增谷歌ComflyGeminiAPI节点，gemini-2.0-flash-exp-image多模态模型。
目前只支持简单的单图输入和输出，图片回复，图片编辑.
<details>
<summary>查看更新/Update </summary>  
 
![微信图片_20250319214344](https://github.com/user-attachments/assets/4ef9216d-1a27-4b71-a5f9-ad4ef4bfc7eb)


</details> 

20250318：`kling节点`: 新增可灵文生视频，图生视频，视频延长（只支持v1.0模型）3个节点.
可灵视频生成时间大概要5-6分钟左右，使用的是 https://ai.comfly.chat 的 api key.
<details>
<summary>查看更新/Update </summary>  
 
![微信图片_20250318201313](https://github.com/user-attachments/assets/96836710-95f7-4100-96ed-58e5d6553124)

</details> 

20250220：`chat按钮`: 修复在ai模块单击选择模型不生效问题。 删除一些节点，简化mj节点组。

20250219：`mj按钮`: 修复在ai模块里面，mj生成图片后点击U1等按钮失效问题。新增：ai模块上方的模型选择可以双击搜索模型功能。
删除了侧边栏helpinfo按钮。
<details>
<summary>查看更新/Update </summary>  
 
![8b8ec1ca909343daae7a0a64b542b54](https://github.com/user-attachments/assets/28e93ac5-558a-49f2-88de-cc8b9151a49c)

</details>  

20241021：`comfly按钮`: 修改悬浮按钮关闭方式，直接按键盘的“A”+“X”按钮关闭和隐藏悬浮按钮，再次按键盘快捷键：“A”+“S”按钮显示悬浮按钮
注意：A不是Alt，而只是键盘的字母键而已！！！！！
<details>
<summary>查看更新/Update </summary>  
 
![1](https://github.com/user-attachments/assets/57c56b10-e9ea-4162-8193-31a52fc6a6fd)


</details>  

20241012：`插件页面`: 增加Re按钮：插件增加查看本地readme按钮。修改一些更新按钮的bug。
<details>
<summary>查看更新/Update </summary>  
  
![e7db5bfab6500542eb994d8dd78baeb](https://github.com/user-attachments/assets/ed800a3d-56ef-427e-8b8d-893117ce2c74)

![7650f3adaee7eae0f0ef8d2b3a97542](https://github.com/user-attachments/assets/ce6f4085-68a6-462c-b7a2-94b3376b226e)

</details>
  

20240913：`comfly按钮`: 修改悬浮按钮，鼠标放置上去可以点击右上角关闭按钮，再次按键盘快捷键：“A”+“S”按钮显示悬浮按钮
<details>
<summary>查看更新/Update </summary>  
  
![058a06b98ea23688ce3cb3e0c41f418](https://github.com/user-attachments/assets/6c6ed9d4-fd82-45e4-b32a-9d7ecce5c3ea)

</details>  



# 主要功能界面：

## Ai Chat and Midjourney

* `Ai Chatbox `: 这个就是悬浮按钮，可以点击展开功能模块

![comfly编辑 (4)](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/ad5b4fde-2953-4706-a528-0d99ad8d62ee)



* `Midjourney `:和各类大语言模型

![comfly编辑 (5)](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/d8656f33-0ea7-4a10-beba-0a44886cf8f4)



* `Midjourney `:的节点组，workflow：


![ER%`0A514D7` 6C3WLQ6)BA](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/e8b559c6-bfd1-4dde-801e-8f49b4e1a897)



> \[!IMPORTANT]\
> 由于ai和 midjourney api需要api key,请直接在这个网址：https://ai.comfly.chat
> 
> api key可以用在任何支持自定义的第三方软件上面，直接在线使用网址：https://ai.comfly.chat/chat
>
> 本插件还带有coze，kling可灵文生图节点，都是免费的，需要自己填写自己的bot api或者cookie即可。
> 
> ai模块有suno还有几个大模型没有适配，现在有点忙，还没有时间测试和适配，后续有时间再说。
>

## api key数据填写文件：Comflyapi.json

![1721054266467](https://github.com/user-attachments/assets/4164b383-090c-4bfe-8c09-f3d0daae0de7)



# 🥵 Comfly的QQ群 / my wechat

![86601b471a343671e7240c74aa8e1fd](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/3e1c2d15-ba5b-4aa5-a76b-08f87e7c8e2c)

![86601b471a343671e7240c74aa8e1fd](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/fdc2f849-5937-4cce-a36d-8444ecca3030)




## :sparkling_heart:依赖管理器
* `Dependencies `: 这个主要是用来管理comfyui安装好的依赖，安装和卸载，包括查看依赖的所有版本，还有环境的冲突有哪些 

![comfly编辑](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/dc1752c8-8d64-4364-9ba3-21507cbaacd8)


  
  
## :tangerine:Comfyui版本管理
* `Comfyui version`: comfyui本体版本的管理和切换.


![comfly编辑 (1)](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/fee00ca2-b4e3-474a-a002-708a05f2adcb)



## :cactus:插件管理
* `Costum nodes`: 管理插件的安装，启用和禁用，插件的版本管理，更新等等.

![comfly编辑 (2)](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/d060808f-7408-4bb5-bd62-981299da79f8)




## :partying_face:插件依赖文件安装和修改
* `Requirements`: 可以查看和修改依赖版本，并且直接一键安装.


![comfly编辑 (3)](https://github.com/ainewsto/Comfyui_Comfly/assets/113163264/8d685533-52cb-4de7-ae8e-3420b6fa804d)




# :dizzy:插件有参考项目 Comfy Resources：

https://github.com/kijai/ComfyUI-KJNodes

感谢原项目：
https://github.com/comfyanonymous/ComfyUI



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ainewsto/Comfyui_Comfly&type=Date)](https://star-history.com/#ainewsto/Comfyui_Comfly&Date)



## 🚀 About me
* website: https://comfly.chat
* Welcome valuable suggestions! 📧 **Email**: [3508432500@qq.com](mailto:1544007699@qq.com)
