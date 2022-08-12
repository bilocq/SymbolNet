"""
Interactive mode for SymbolNet
"""

import os
import argparse
import pickle
from copy import copy
import torch
from torchvision import transforms
from PIL import Image, ImageGrab, ImageOps, ImageTk
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

from symbolnet_utils import networks





# -------------------------- USEFUL FUNCTIONS --------------------------
#########################
### Drawing on canvas ###
#########################
def draw(event):
    color = 'white'
    boldness = 15
    x1, y1 = (event.x - boldness), (event.y - boldness)
    x2, y2 = (event.x + boldness), (event.y + boldness)
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)

def erase(event):
    color = 'black'
    boldness = 10
    x1, y1 = (event.x - boldness), (event.y - boldness)
    x2, y2 = (event.x + boldness), (event.y + boldness)
    canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)

def clear_canvas():
    canvas.delete('all')

def canvas_enable():
    canvas_title.config(text="Draw you symbol here!")
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<Control-B1-Motion>', erase)  

def canvas_disable():
    canvas_title.config(text="")
    canvas.unbind('<B1-Motion>')
    canvas.unbind('<Control-B1-Motion>')
    

###############################
### Extracting canvas image ###
###############################
def get_canvas():
    """
    Gets canvas image, changes it to grayscale mode
    """
    img = ImageGrab.grab(bbox=(canvas.winfo_rootx()+5,
                               canvas.winfo_rooty()+5,
                               canvas.winfo_rootx() + canvas.winfo_width() - 5,
                               canvas.winfo_rooty() + canvas.winfo_height() - 5))
    img = ImageOps.grayscale(img)
    return img

def get_canvas_lowres():
    """
    Gets canvas image, resizes it to 28x28 (EMNIST size) and makes a tensor version of it. Returns low resolution image
    and tensor version. Also checks if the tensor is identically zero and returns a boolean indicator.
    """
    img = get_canvas()
    img = img.resize((28,28), resample=Image.Resampling.BICUBIC)
    tensor_img = to_tensor(img)
    all_zeros = torch.all(tensor_img == torch.zeros(tensor_img.shape)) # True iff tensor_img is identically zero
    return img, tensor_img, all_zeros    


###############################
### Low resolution displays ###   
###############################     
def display_lowres(img, lowres_holder, spot):
    """
    Displays the low resolution version of the canvas image that is given to the network.
    """
    img = img.resize((112,112))
    lowres_holder.image = ImageTk.PhotoImage(img)
    lowres_holder.config(image=lowres_holder.image)
    lowres_arrow.create_line(0, 10, 150, 10, arrow=tk.LAST)
    lowres_caption.config(text=arrow_caption)
    if spot == 1:
        lowres_holder.place(x=585, y=25, width=112, height=112)
    elif spot == 2:
        lowres_holder.place(x=712, y=25, width=112, height=112)
    else:
        raise ValueError('display_lowres given invalid spot. Options for spot argument are 1 and 2.')

def remove_display():
    lowres_arrow.delete('all')
    lowres_caption.config(text='')
    lowres_holder_1.config(image='')
    lowres_holder_2.config(image='')


################
### Text box ###
################
def replace_text(message):
    SN_text['state'] = 'normal'
    SN_text.delete('1.0', 'end')
    SN_text.insert('1.0', message)
    SN_text['state'] = 'disabled'

def bold_text(start, end):
    bold_font = tkfont.Font(SN_text, SN_text.cget('font'))
    bold_font.config(weight='bold')
    SN_text.tag_configure('bold', font=bold_font)
    SN_text.tag_add('bold', start, end)
    
def small_text(start, end):
    small_font = tkfont.Font(SN_text, SN_text.cget('font'))
    small_font.config(size=10)
    SN_text.tag_configure('small', font=small_font)
    SN_text.tag_add('small', start, end)

def big_text(start, end):
    big_font = tkfont.Font(SN_text, SN_text.cget('font'))
    big_font.config(size=30)
    SN_text.tag_configure('big', font=big_font)
    SN_text.tag_add('big', start, end)


#################
### Entry box ###
#################
def entry_enable():
    entry['state'] = 'normal'
    entry_caption.config(text="Write class name and press enter")

def entry_disable():
    entry['state'] = 'disabled'
    entry_caption.config(text='')
    

#################   
### Resetting ###
#################     
def reset_buttons():
    """
    Returns all buttons to normal state
    """
    cb1.config(text='Compare to another drawing', command=compare_ims_1)
    cb2.config(text='Guess class', command=guess_class)
    cb3.config(text='Teach a new class', command=teach_class_start)
    tb1.config(text='Tell me about you', command=explain_1)
    tb2.config(text="What can I do here?", command=explain_2)
    tb3.config(text='')
    tb4.config(text='') 
    cb1['state'] = 'normal'
    cb2['state'] = 'normal'
    cb3['state'] = 'normal'
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'disabled'
    tb4['state'] = 'disabled'
    if memory_modified:
        mb1['state'] = 'normal'
        mb2['state'] = 'normal'

def reset():
    """
    Returns everything to initial state except displays, text box, network memory and memory buttons
    """
    canvas_enable()
    entry_disable()
    clear_canvas()
    reset_buttons()
    # Reset global variables
    global img_tensor_1, img_tensor_2, sim_scores
    global images_to_mem, tensors_to_mem
    global class_name, known_class
    img_tensor_1, img_tensor_2, sim_scores = None, None, None
    images_to_mem, tensors_to_mem = None, None
    class_name, known_class = None, None

def Reset(*event):
    """
    Returns everything to initial state except text box, network memory and memory buttons
    """
    reset()
    remove_display()

def RESET(*event):
    """
    Returns everything to initial state except network memory and memory buttons
    """
    Reset()
    replace_text('Hello, my name is SymbolNet.')
  
  
#################  
### Reactions ###
#################
def verdict_reaction_1(message):
    replace_text(message)
    big_text('1.0', '1.1')
    reset()

def verdict_reaction_2(message):
    replace_text(message)
    big_text('1.0', '1.1')
    canvas_enable()
    tb1['state'] = 'disabled'
    tb2['state'] = 'disabled'
    cb1['state'] = 'normal'
    cb2['state'] = 'normal'
    cb3['state'] = 'normal'
    
def verdict_reaction_3(img, img_tensor):
    message = 'You can teach me a new class starting with this image!'
    replace_text(message)
    tb1.config(text='Teach new class', command = lambda: teach_class_start(img, img_tensor))
    tb2['state'] = 'disabled'
    
def empty_canvas():
    """
    Invoked when an empty canvas is submitted. Tells user to draw something.
    """
    SN_text['state'] = 'normal'
    message_1 = "You have to draw something!\n"
    message_2 = "I'm not interested in looking at empty canvases!\n"
    message_3 = "\N{expressionless face}\n"
    # Get current text to adjust message, then delete it
    previous_message = SN_text.get('1.0', 'end')
    SN_text.delete('1.0', 'end')
    # Write message
    new_message = message_1
    if message_1 in previous_message:
        new_message += message_2
    if message_2 in previous_message:
        new_message += message_3
    SN_text.insert('end', new_message)
    SN_text['state'] = 'disabled'


#####################
### Using network ###
#####################
@torch.no_grad()
def evaluate_pair():
    """
    Evaluates similarity score between img_tensor_1 and img_tensor_2
    """
    t1 = img_tensor_1.unsqueeze(0)
    t2 = img_tensor_2.unsqueeze(0)
    out = softmax(net(t1, t2))
    return out

@torch.no_grad()
def classify_image(img):
    """
    Uses memory to classify 'img'.
    """
    similarity_scores = []
    mem_dict = net.memory_dict
    for class_ in list(mem_dict.keys()):
        class_mem = mem_dict[class_]
        num_mem = next(iter(class_mem.values())).shape[0] # Find number of memorized images for class
        im_feats = net.feature_network(torch.stack([img]*num_mem)) # Get features from image
        sim_score = torch.mean(softmax(net.evaluator(im_feats, class_mem))[:,1]).numpy()
        similarity_scores.append((class_, sim_score))
    similarity_scores.sort(key=lambda x:x[1], reverse=True)
    return similarity_scores

@torch.no_grad()
def get_features(imgs):
    return net.feature_network(imgs)





# -------------------------- MAIN FLOW --------------------------
###################
### Information ###
###################
def explain_1():
    # Message 
    message  =  "I'm a neural network designed to judge whether a pair of handdrawn images are from the same 'class'. When "
    message +=  "shown a pair of images, I output a number between 0 and 1 called a 'similarity score'. The larger this "
    message +=  "score is, the more likely I think it is that the two images belong to the same class. \n\n"   
    message +=  "The meaning of a 'class' of images is context dependent. I was trained in the context of handwritten symbols "
    message +=  "(hence my name). Therefore, for me, the letter 'A' is a class and the letter 'B' is another class. If you show "
    message +=  "me a handwritten A along with a handwritten B, I should output a low similarity score; but if you show me two "
    message +=  "handwritten A's, I should output a high similarity score. More specifically, I was trained by looking at a total "
    message += f"of {num_train_samples} pairs of images taken from the following classes:\n\n"
    for c in mem_list:
        message += f'{c}  '
    message += "\n\nand by guessing for each pair whether the images were from the same class or not.\n\n"
    message += "This training method allowed me to learn what a 'class' of images means in the context of handwritten symbols. "
    message += "An interesting consequence of this is that I can usually still determine whether two handwritten symbols are "
    message += "from the same class when they are from classes that have not been used in my training! For instance, you "
    message += "could ask me to compare an 'alpha' to a 'beta' and I should conclude that they are from different classes "
    message += "(hopefully). Similarly, you could show me, say, two handwritten question marks and I should conclude that they "
    message += "are from the same class. This is an example of 'transfer learning'.\n\n"
    message += "DISCLAIMER\n"
    message += "I'm only a prototype, and my performance is certainly not close to any sort of state-of-the-art level. "
    message += "I'm likely to give you some wonky answers from time to time!"
    replace_text(message)
    # Make some part bold
    bold_text('1.6', '1.20')
    bold_text('1.174', '1.190')
    bold_text('3.0', '3.56')
    bold_text('9.240', '9.307')
    bold_text('9.590', '9.610')
    bold_text('11.0', '11.10')
    # Adjust buttons
    tb1.config(text="Classification?", command=explain_2)
    tb2.config(text="What can I do here?", command=what_to_do)
    tb3.config(text='What symbols do you know?', command=show_memory)
    tb4.config(text='Ok!', command=RESET)
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'normal' 
    tb4['state'] = 'normal'
            
def explain_2():  
    message  = "Being able to judge the similarity of pairs of images allows me to classify new images into classes I've "  
    message += "previously memorized! For example, by keeping a number of handwritten A's in my memory, I can then judge "
    message += "whether or not a new image is likely to be an A by comparing this new image to the memorized A's. More "
    message += "generally, by memorizing examples from various classes and comparing a new image to each of these classes, I "
    message += "can classify the new image in the class that gets the highest average similarity score, or decide that the "
    message += "new image does not belong to any of the classes I have memorized.\n\n"
    message += "You can teach me to recognize a new class simply by showing me examples from that class. If you want me to "
    message += "be able to classify greek letters, you can show me handwritten examples of each greek letter and ask me "
    message += "to memorize them. Even though I was never shown greek letters during my training, I should then be able to "
    message += "classify greek letters reasonably well, with no extra training! Note that this is different from a standard  "
    message += "classification neural network, where the classes are rigidly embedded in the architecture (they correspond " 
    message += "to 'logits', i.e. the outputs of the last layer)."
    replace_text(message)
    # Make some parts bold
    bold_text('1.67', '1.76')
    bold_text('1.116', '1.125')
    bold_text('3.8', '3.29')
    # Adjust buttons
    tb1.config(text="Tips", command=explain_3)
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'normal' 
    tb4['state'] = 'normal'

def explain_3():
    message  = "TIPS:\n"
    message += "  - The inputs used during my training contained a single symbol each, so my performance is likely to be "
    message += "better if you show me single symbols (e.g. letters, digits, etc), instead of e.g. drawings or strings of "
    message += "symbols. But, of course, you are free to try drawing whatever you want.\n"
    message += "  - The drawing functionality is deliberately set to a bold font to reflect the images that were used during "
    message += "my training.\n"
    message += "  - I'm used to seeing images with relatively well centered symbols (no data augmentation was used during my "
    message += "training), so it's likely that I'll perform better if you try to center your drawings in the canvas.\n\n"
    replace_text(message)
    # Make some parts bold
    bold_text('1.0', '1.4')
    bold_text('6.0', '6.10')
    # Adjust buttons
    tb1.config(text='Tell me more about you', command=explain_1)
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'normal' 
    tb4['state'] = 'normal'

def what_to_do():
    # Message
    message  = "Here, you can draw symbols and show them to me. The three buttons on the bottom left let you decide how you want me "
    message += "to process these symbols.\n\n"
    message += "  - You can show me a pair of symbols drawn in sequence, and ask me whether I think they belong to the same class.\n\n"
    message += "  - You can show me a symbol and ask me to guess what it is. I will then compare your symbol to all the classes I "
    message += "know and compute an average similarity score for each of these classes. I will let you know what class(es) match your "
    message += "symbol, if I find any.\n\n"
    message += "  - You can teach me a new class, or change the memory I have of a class I already know."
    replace_text(message)
    # Adjust buttons
    tb3.config(text='What symbols do you know?', command=show_memory)
    tb4.config(text='Ok!', command=RESET)
    tb1['state'] = 'normal'
    tb2['state'] = 'disabled'
    tb3['state'] = 'normal' 
    tb4['state'] = 'normal'

def show_similarities():  
    SN_text['state'] = 'normal'
    message = '\n\nHere are the similarity scores for all classes:\n'
    for class_, score in sim_scores:
        message += f"{class_}: {score:.2f}    "
    SN_text.insert('end', message)
    SN_text['state'] = 'disabled'
    # Adjust buttons
    tb3['state'] = 'disabled'
    
def show_memory():
    message = "These are the classes I currently have memorized:\n\n"
    for c in net.memory_dict.keys():
        message += f"{c}  "
    message += "\n\nThis means that if you ask me to guess the class of a symbol, I can only guess one of these classes. However, "
    message += "you can teach me new classes here!\n\n"
    message += "If you find that I am bad at guessing a class in particular, it may be because I have memorized examples from this "
    message += "class where the handwritting was very different from yours. You can override my memory of this class here, by using "
    message += "the 'Teach me a new class' button and entering the name of this class."
    if net.new_mem: # There are new classes in memory dict
        message += "\n\nSince the start of this session, you have taught me the following new classes:\n\n"
        for c in net.new_mem:
            message += f"{c}  "
    if net.modified_mem: # There are modified classes in memory dict
        message += "\n\nSince the start of this session, you have modified my memory of the following classes:\n\n"
        for c in net.modified_mem:
            message += f"{c}  "
    replace_text(message)
    # Adjust buttons
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'disabled' 
    tb4['state'] = 'normal'


########################
### Comparing images ###
########################
def compare_ims_1():
    global img_tensor_1
    img, img_tensor_1, all_zeros = get_canvas_lowres()
    clear_canvas()
    remove_display()
    if all_zeros: # Check if canvas is empty
        empty_canvas()
        return
    display_lowres(img, lowres_holder_1, spot=1)
    message = "Show me another symbol, and I'll guess whether it's in the same class as this one."
    replace_text(message)
    # Adjust buttons
    cb2['state'] = 'disabled'
    cb3['state'] = 'disabled'
    tb1['state'] = 'disabled'
    tb2['state'] = 'disabled'
    tb3['state'] = 'disabled'
    tb4['state'] = 'normal'
    cb1.config(text='Compare to first drawing', command=compare_ims_2)
    tb1.config(text='')
    tb2.config(text='')
    tb3.config(text='')
    tb4.config(text='Reset', command=RESET)
    if memory_modified:
        mb1['state'] = 'normal'
        mb2['state'] = 'normal'
      
def compare_ims_2():
    global img_tensor_1, img_tensor_2, out
    img, img_tensor_2, all_zeros = get_canvas_lowres()
    clear_canvas()
    if all_zeros: # Check if canvas is empty
        empty_canvas()
        return
    display_lowres(img, lowres_holder_2, spot=2)
    out = evaluate_pair()[0]
    # Discuss results
    if out[1] >= net.threshold: # Same classes:
        message = f'I think these are the same.\nSimilarity score: {out[1]:.3f}'
    else:
        message = f'I think these are different.\nSimilarity score: {out[1]:.3f}'
    message += f'\n\n\n(The current similarity threshold is {net.threshold}. You can change it manually by '
    message +=  'closing this window and changing the net.threshold parameter near the end of the '
    message +=  'interactive.py script. Making the threshold larger will make me less likely to guess that '
    message +=  'two images are in the same class.'
    replace_text(message)
    small_text('3.0', tk.END)
    # Adjust buttons and canvas
    img_tensor_1, img_tensor_2 = None, None
    canvas_disable()
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    cb1['state'] = 'disabled'
    cb1.config(text='Compare to another drawing')
    tb1.config(text='You are right!', command = lambda: verdict_reaction_1("\N{grinning face}"))
    tb2.config(text='You are wrong!', command = lambda: verdict_reaction_1("\N{crying face}"))
          
   
######################
### Guessing class ###
######################    
def guess_class():
    global sim_scores 
    remove_display()
    img, img_tensor, all_zeros = get_canvas_lowres()
    clear_canvas()
    if all_zeros: # Check if canvas is empty
        empty_canvas()
        return 
    display_lowres(img, lowres_holder_1, spot=1)
    sim_scores = classify_image(img_tensor)
    # Numbers of classes above threshold and above threshold
    num_thresh = next((i for i in range(len(sim_scores)) if sim_scores[i][1] < net.threshold), len(sim_scores)) 
    # Discuss results
    if num_thresh == 1:
        message  = f"I am confident that this is a {sim_scores[0][0]}, since that's the only class with a similarity "
        message += f"score above the {net.threshold} threshold.\n\n"
        message += f"{sim_scores[0][0]}   (similarity score: {sim_scores[0][1]:.2f})"
        tb1.config(text="You are right!", command = lambda: verdict_reaction_2("\N{grinning face}"))
        tb2.config(text="You are wrong!", command = lambda: verdict_reaction_2("\N{crying face}"))
    elif num_thresh > 1:
        message  = "Hmm, I think this could be one of the following:\n\n"
        for i in range(num_thresh):
            message += f"{sim_scores[i][0]}   (similarity score: {sim_scores[i][1]:.2f})\n"
        tb1.config(text="That's true", command = lambda: verdict_reaction_2("\N{relieved face}"))
        tb2.config(text="That's not true", command = lambda: verdict_reaction_2("\N{loudly crying face}"))
    else:
        message  = f"None of the classes I currently have memorized gets a similarity score above the {net.threshold} "
        message +=  "threshold. I don't know what this symbol is."
        tb1.config(text="It's a class you don't know", command = lambda: verdict_reaction_3(img, img_tensor))
        tb2.config(text="You should know!", command = lambda: verdict_reaction_2("\N{unamused face} ...sorry..."))
    replace_text(message)
    # Adjust buttons and canvas
    canvas_disable()
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'normal'
    tb4['state'] = 'normal'
    cb1['state'] = 'disabled'
    cb2['state'] = 'disabled'
    cb3['state'] = 'disabled'
    tb3.config(text="Show all similarity scores", command=show_similarities)
    tb4.config(text="Reset", command=RESET)
    if memory_modified:
        mb1['state'] = 'normal'
        mb2['state'] = 'normal'
    

##########################
### Teaching new class ###
##########################
def teach_class_prompt():
    message = "You can teach me!"
    replace_text(message)

def teach_class_start(img=None, img_tensor=None):
    global images_to_mem, tensors_to_mem
    if (img is None) or (img_tensor is None):
        img, img_tensor, all_zeros = get_canvas_lowres()
    else:
        all_zeros = False
    clear_canvas()
    remove_display()
    # Lists of what is to be memorized
    images_to_mem  = [img]
    tensors_to_mem = [img_tensor]
    if all_zeros: # Check if canvas is empty
        empty_canvas()
        return
    display_lowres(img, lowres_holder_1, spot=1)
    entry_enable()
    # Message
    message =  "I'm ready to learn!\n\n"
    message += "The first thing you have to do is write a class name below and press 'enter'. For instance, you could "
    message += "write 'alpha' if you want to teach me to recognize the greek letter alpha.\n\n"
    message += "  - If you write a new class name (i.e. one I don't currently know), then I will learn this new symbol and "
    message += "take it into consideration when you ask me to classify images later.\n\n"
    message += "  - If you write a class name that I do currently know, then I will override my memory of that class.\n\n"
    message += "Here are the names of the classes that I currently know:\n"
    for c in net.memory_dict.keys():
        message += f'{c} '
    replace_text(message)
    # Adjust buttons and canvas
    canvas_enable()
    tb1['state'] = 'disabled'
    tb2['state'] = 'disabled'
    tb3['state'] = 'disabled'
    tb4['state'] = 'disabled'
    cb1['state'] = 'disabled'
    cb2['state'] = 'disabled'
    cb3['state'] = 'disabled'
    tb1.config(text="Commit memory", command=commit_memory)
    tb2.config(text="Change class name", command=change_class_name)
    tb3.config(text="Show images", command=show_mem_imgs)
    tb4.config(text="Cancel", command=RESET)
    cb3.config(text="Add image", command=memorize_canvas)

def teach_class_main(*event): 
    global class_name, known_class
    class_name = entry.get()
    entry_disable()
    known_class = (class_name in net.memory_dict.keys()) # Check whether class is already memorized
    # Message
    if known_class:
        message = f"'{class_name}' is a symbol that I already know, but now I will override my memory of it.\n\n"
    else:
        message = f"I'm now learning a new symbol named '{class_name}'.\n\n"
    message += f"Number of examples of {class_name}'s you have shown to me up to now: {len(images_to_mem)}\n\n"
    if len(images_to_mem) < 5:
        message += f"You are encouraged to show me a few more examples of {class_name}'s, by simply drawing them and using the " 
        message +=  "bottom left button. Try to vary your handwriting style, or ask a friend to come draw some of the examples! "
        message +=  "When you feel you've shown me enough examples, press the 'Memorize' button. This will prompt me to memorize "
        message += f"these examples under the class name {class_name}. You can also use the 'Show images' button to see all the "
        message +=  "images you have submitted for me to memorize."
    replace_text(message)
    # Adjust buttons
    tb1['state'] = 'normal'
    tb2['state'] = 'normal'
    tb3['state'] = 'normal'
    tb4['state'] = 'normal'
    cb1['state'] = 'disabled'
    cb2['state'] = 'disabled'
    cb3['state'] = 'normal'
    tb1.config(text="Memorize", command=commit_memory)
    tb2.config(text="Change class name", command=change_class_name)
    tb3.config(text="Show images", command=show_mem_imgs)
    tb4.config(text="Cancel", command=RESET)
    cb3.config(text="Add image", command=memorize_canvas)
    if memory_modified:
        mb1['state'] = 'normal'
        mb2['state'] = 'normal'

def memorize_canvas():
    global images_to_mem, tensors_to_mem
    img, img_tensor, all_zeros = get_canvas_lowres()
    if all_zeros: # Check if canvas is empty
        empty_canvas()
        return
    clear_canvas()
    images_to_mem.append(img)
    tensors_to_mem.append(img_tensor)
    # Display last two submitted images in reverse order
    display_lowres(img, lowres_holder_1, 1)
    display_lowres(images_to_mem[-2], lowres_holder_2, 2)
    # Adapt message
    teach_class_main()

def change_class_name():
    message = "Enter the new class name"
    replace_text(message)
    entry.delete('0', 'end')
    entry['state'] = 'normal'
    # Adjust buttons
    tb1['state'] = 'disabled'
    tb2['state'] = 'disabled'
    tb3['state'] = 'disabled'
    tb4['state'] = 'disabled'
    cb3['state'] = 'disabled'    

def show_mem_imgs():
    num_ims = len(images_to_mem)
    # Message
    if num_ims == 0:
        message = "\n\nYou haven't shown me any images to memorize yet!"
    elif num_ims == 1:
        message = f"\n\nHere is the single example of a {class_name} that you want me to memorize: \n\n"
    else:
        message = f"\n\nHere are the examples of {class_name}'s that you want me to memorize: \n\n"
    SN_text['state'] = 'normal'
    SN_text.delete('4.0', tk.END)
    SN_text.insert(tk.END, message)
    # Add images to text box
    global images_to_mem_ # Keep reference to avoid PIL garbage collection bug
    images_to_mem_ = []
    for i in range(num_ims):
        # Gray padding
        if (i < 15*((num_ims-1)//15)):          # Not in the last line
            if (i < num_ims-1) and ((i+1)%15 != 0):      # Not rightmost -> pad to the right and below
                im = Image.new('L', (30,30), color=100)
                im.paste(images_to_mem[i], (0,0))
            else:                                        # Rightmost -> pad below only
                im = Image.new('L', (28,30), color=100)
                im.paste(images_to_mem[i], (0,0))
        else:                                   # In the last line
            if (i < num_ims-1):                          # Not rightmost -> pad to the right
                im = Image.new('L', (30,28), color=100)
                im.paste(images_to_mem[i], (0,0))
            else:                                        # Rightmost -> don't pad at all
                im = images_to_mem[i]
        images_to_mem_.append(ImageTk.PhotoImage(im))
        SN_text.image_create(tk.END, image=images_to_mem_[i])
    SN_text['state'] = 'disabled'
    # Adjust buttons 
    tb3['state'] = 'disabled'

def commit_memory():
    global features, memory_modified
    memory_modified = True
    if known_class:
        net.modified_mem.append(class_name)
    else:
        net.new_mem.append(class_name)
    mem_tensor = torch.stack(tensors_to_mem, dim=0)
    net.memory_dict[class_name] = get_features(mem_tensor)
    message = f"Great! Now that I've memorized what {class_name}'s look like, I should be able to recognize that symbol."
    replace_text(message)
    reset()
   
def reset_memory_1():
    if memory_file_rewritten:
        message  = "You can reset my memory either to what it was the last time you saved or to what it was at the "
        message += "start of this session (before you saved my new memory)."
        replace_text(message)
        tb2['state'] = 'normal'
        tb1.config(text="Reset to last save", command=reload_memory)
        tb2.config(text="Reset to session start", command=reset_memory_start)
    else:
        message = "Resetting my memory means it'll go back to what it was when this session started."
        replace_text(message)
        # Adjust buttons
        tb2['state'] = 'disabled'
        tb1.config(text="Confirm", command=reset_memory_start)
    tb1['state'] = 'normal'
    tb3['state'] = 'disabled'
    tb4['state'] = 'normal'
    cb1['state'] = 'disabled'
    cb2['state'] = 'disabled'
    cb3['state'] = 'disabled'
    mb1['state'] = 'disabled'
    mb2['state'] = 'disabled'
    tb4.config(text="Cancel", command=RESET)

def save_memory_1():
    # Message
    message  = f"Saving the memory will change the {mem_file} file so it contains my current memory with the additions and "
    message +=  "changes we have made since the start of this session.\n\n"
    if memory_file_rewritten:
        message += "As a reminder, since the last time my memory was saved, you have:\n"
    else:
        message += "As a reminder, since the start of this session you have:\n"
    if net.new_mem:
        message += "  - Taught me the following new classes:  "
        for c in net.new_mem:
            message += f"'{c}' "
    if net.modified_mem:
        message += "  - Modified my memory of the following classes:  "
        for c in net.modified_mem:
            message += f"'{c}' "
    replace_text(message)
    # Adjust buttons
    tb3['state'] = 'disabled'
    tb4['state'] = 'disabled'
    cb1['state'] = 'disabled'
    cb2['state'] = 'disabled'
    cb3['state'] = 'disabled'
    mb2['state'] = 'disabled'
    tb1.config(text="Confirm", command=save_memory_2)
    tb2.config(text="Cancel", command=RESET)
    
def save_memory_2():
    global memory_modified, memory_file_rewritten
    with open(mem_file, 'wb') as f:
        pickle.dump(net.memory_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    net.new_mem = []
    net.modified_mem = []
    memory_modified = False
    memory_file_rewritten = True
    message = f"My current memory has been save in {mem_file}"
    replace_text(message)
    Reset()

def reload_memory():
    with open(mem_file, 'rb') as f:
        net.memory_dict = pickle.load(f)
    message = "My memory is back to what it was the last time you saved it."
    replace_text(message)
    Reset()

def reset_memory_start():
    net.memory_dict = init_mem_dict
    message = "My memory is back to what it was at the start of this session."
    replace_text(message)
    mb2['state'] = 'normal'
    Reset()





# -------------------------- SET UP WINDOW --------------------------
###################
### Main window ###
###################
root = tk.Tk()
root.title("SymbolNet")
root.geometry('940x600')
root.minsize(500, 600)
root.maxsize(1000, 1000)
 

##############   
### Canvas ###
##############  
# Drawing canvas
canvas = tk.Canvas(root, bg='black')
canvas.place(x=10, y=30, height=400, width=400)
# Canvas title
title_font = tkfont.Font(size=16, weight='bold')
canvas_title = ttk.Label(root, text="Draw you symbol here!", font=title_font, anchor=tk.CENTER)
canvas_title.place(x=10, y=0, height=30, width=400)
# Canvas label
message = "press and drag to draw.\n"
message += "ctrl + press and drag to erase."
canvas_caption = ttk.Label(root, text=message)
canvas_caption.place(x=20, y=430, width=200)
# Enable
canvas_enable()
  
 
###############################
### Low resolution displays ###
###############################
# Low resolution image
lowres_holder_1 = ttk.Label(root)
lowres_holder_2 = ttk.Label(root)
lowres_arrow = tk.Canvas(root)
lowres_caption = ttk.Label(root)
arrow_font = tkfont.Font(size=9)
arrow_caption = "Low resolution version\nshown to network"
lowres_caption.config(text='', font=arrow_font, anchor=tk.CENTER)
lowres_arrow.place(x=420, y=95, height=20, width=160)
lowres_caption.place(x=420, y=55, height=50, width=140)

    
################    
### Text box ###
################
# SymbolNet text box
text_font = tkfont.Font(size=13)
SN_text = tk.Text(root, height=5, width=60, padx=20, pady=10, font=text_font)
SN_text.place(x=425, y=160, height=200, width=500)
SN_text.config(highlightbackground = "blue", wrap=tk.WORD)
SN_text.insert('1.0', 'Hello, my name is SymbolNet.\n')
scrollbar = ttk.Scrollbar(root, orient='vertical', command=SN_text.yview)
scrollbar.place(x=905, y=170, height=180)
SN_text['yscrollcommand'] = scrollbar.set
SN_text['state'] = 'disabled'


###############
### Buttons ###
###############
# Canvas clear button
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.bind('<Double-Button-1>', RESET)
clear_button.place(x=230, y=430, width=100, height=35)
# Canvas buttons
cb1 = tk.Button(root, text="Compare to another drawing", command=compare_ims_1)
cb2 = tk.Button(root, text="Guess class", command=guess_class)
cb3 = tk.Button(root, text="Teach a new class", command=teach_class_start)
cb1.place(x=75, y=470, width=250, height=30)
cb2.place(x=75, y=500, width=250, height=30)
cb3.place(x=75, y=530, width=250, height=30)
# Text buttons
tb1 = tk.Button(root, text="Tell me about you", command=explain_1)
tb2 = tk.Button(root, text="What can I do here?", command=what_to_do)
tb3 = tk.Button(root)
tb4 = tk.Button(root)
tb3['state'] = 'disabled'
tb4['state'] = 'disabled'
tb1.place(x=425, y=370, width=250, height=30)
tb2.place(x=685, y=370, width=250, height=30)  
tb3.place(x=425, y=400, width=250, height=30)  
tb4.place(x=685, y=400, width=250, height=30)  
# Memory buttons
mb1 = tk.Button(root, text="Reset memory", command=reset_memory_1)
mb2 = tk.Button(root, text="Save memory", command=save_memory_1)
mb1['state'] = 'disabled'
mb2['state'] = 'disabled'
mb1.place(x=600, y=500, width=150, height=30)
mb2.place(x=600, y=530, width=150, height=30)


##################
### Entry line ###
##################
entry = tk.Entry(root)
entry.place(x=550, y=440, width=250, height=30)
entry.bind('<Return>', teach_class_main)
entry['state'] = 'disabled'
entry_font = tkfont.Font(size=11, weight='bold')
entry_caption = ttk.Label(root, text='', font=entry_font)
entry_caption.place(x=585, y=470, height=20, width=250)





# -------------------------- SETTING UP NETWORK --------------------------
###########################################
### Get folder as command line argument ###
###########################################
path = './pretrained_example'
parser = argparse.ArgumentParser('...')
parser.add_argument('--folder', default=path, help="Folder containing the SymbolNet network to test.")
args = parser.parse_args()
folder = args.folder


########################
### Misc definitions ###
########################
device = torch.device('cpu') # No need for GPU here
to_tensor = transforms.ToTensor() # Used to transform canvas drawing into tensor
softmax = torch.nn.Softmax(dim=1) # Used to get similarity scores from network ouputs


####################
### Load network ###
####################
with open(os.path.join(folder, 'network_info.pkl'), 'rb') as net_dict_file:
    net_dict = pickle.load(net_dict_file)
arch              = net_dict['architecture']
comp_mode         = net_dict['compare_mode']
blocks            = net_dict['blocks']
mem_list          = net_dict['training_classes']
num_train_samples = net_dict['num_training_samples']
arch   = getattr(networks, arch)
blocks = [int(b) for b in blocks.split(',')]
net = arch(blocks=blocks, compare_mode=comp_mode).to(device)
net.load_state_dict(torch.load(os.path.join(folder, 'network_state_dict.pth'), map_location=device))
# Load network memory and save it as a network attribute
mem_file = os.path.join(folder, 'memory.pkl')
assert os.path.isfile(mem_file), f"'memory.pkl' missing from {folder}"
with open(mem_file, 'rb') as f:
    init_mem_dict = pickle.load(f) # Dictionary of examples of features corresponding to each class
net.memory_dict = copy(init_mem_dict)
net.new_mem = []
net.modified_mem = []
memory_modified = False
memory_file_rewritten = False
# Threshold for confident class guesses
net.threshold = 0.85





# -------------------------- ACTIVATE --------------------------
root.mainloop()