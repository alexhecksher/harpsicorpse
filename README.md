# harpsicorpse
Haprsicorpse (the full body instrument)

# About
The harpsicorpse is a musical instrument I built that is controlled by body parts and images. 
The original idea was to make an instrument that responded to human bodies on screen 
so that I could feed in movie scenes and generate a score from the movements, gestures, and expressions from the people within the scene. 
I was interested in how to translate the visually expressive human body into a musically expressive instrument.

Check out the harpsicorpse page on my website for more: [(Harpsicorpse)](https://alexhecksher.com/portfolio/harpsicorpse/)

# Setting the harpsicorpse
First, you must have python3 and supercollider installed. 
The version of python3 I am using is Python 3.11.7. The version of supercollider I am using is 3.13.0.

Next, install the necessary python libraries that are listed in requirements.txt.
You can run 'python3 -m pip install -r requirements.txt

# Running the harpsicorpse
To run the instrument you will need to launch and run the supercollider file. Then run the python script.

You can run the script in the other order (python, then supercollider), but you may get a few warnings in supercollider
as it will try to create the synths before they are defined. However, as long as you run both scripts, you should be fine.
