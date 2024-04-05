// Alex Hecksher (2024)
// Code is free to copy and use for non-commericial uses as long as it is properly attributed
// to me and the other people, whose code and knowledge I am used in creating this harpsicorpse.
// Please reach out to me (alexhecksher@gmail.com) if you have any questions
//
// Sections of this code were written with the help of or copied directly from the following sources:
//   - "Tensorflow Multi-Person Pose Estimation with Python // Machine Learning Tutorial"
//       by Nicholas: https://www.youtube.com/watch?v=KC7nJtBHBqg
//   - The mediapipe guide : https://developers.google.com/mediapipe/solutions/guide
//   - "SuperCollider Tutorial: 21. FM Synthesis, Part I" by Eli Fieldsteel:
//       https://www.youtube.com/watch?v=UoXMUQIqFk4&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=22
//   - "SuperCollider Tutorial: 22. FM Synthesis, Part II" by Eli Fieldsteel:
//       https://www.youtube.com/watch?v=dLMSR2Kjq6Y&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=23
//   - "SuperCollider Tutorial: 23. Wavetable Synthesis, Part I" by Eli Fieldsteel:
//       https://www.youtube.com/watch?v=8EK9sq_9gFI&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=24
//   - "SuperCollider Tutorial: 24. Wavetable Synthesis, Part II" by Eli Fieldsteel:
//       https://www.youtube.com/watch?v=7nrUBbmY1hE&list=PLPYzvS8A_rTaNDweXe6PX4CXSGq4iEWYC&index=25
//   - "Delays, Reverbs, Harmonizers - Week 7 Spring 2021 MUS 499C - Intermediate SuperCollider"
//       by Eli Fieldsteel: https://www.youtube.com/watch?v=eEyYFt3sIWs&list=PLPYzvS8A_rTbTAn-ZExGuVFZgVMwYi1kJ&index=7
//
// Huge thanks to Eli Fieldsteel in particular, whose tutorials on supercollider have been instrumental
// in helping me build this instrument. Here is a link to his youtube page: https://www.youtube.com/@elifieldsteel

// Also please excuse any typos and feel free to bring them to my attention for correcting :)

// Run command + return inside the parentheses to start the harpsicorpse
(
// Close any open windows
Window.closeAll;

// Global Variables Setup
~tempo = 120/60;

// Face varialbes
~faceData = [];
~faces = [];
~faceLMSize = 2;
~faceLMNum = 11;
~faceSize = ~faceLMSize * ~faceLMNum + 1;
~faceScale = (0,2..12);
~minFaceAmp = 0.01;
~maxFaceAmp = 0.07;
~minFaceDur = 0.5;
~faceNum = 0;

// Hand varialbes
~handData = [];
~hands = [];
~handLMSize = 2;
~handLMNum = 6;
~handSize = ~handLMSize * ~handLMNum + 1;
~minHandAmp = 0.01;
~maxHandAmp = 0.07;
~handScale = (0,2..24);
~minHandDur = 1;
~handNum = 0;

// Pose varialbes
~poseData = [];
~poses = [];
~poseLMSize = 2;
~poseLMNum = 10;
~poseSize = ~poseLMSize * ~poseLMNum + 1;
~poseDurScale = Array.geom(5, 4, 0.5);
~maxPoseAmp = 0.1;
~minPoseAmp = 0.01;
~minPoseDur = 0.25;
~poseNum = 0;

// Image varialbes
~imageData = [];
~image = 0;
~imageLMSize = 3;
~imageLMNum = 9;
~imageSize = ~imageLMSize * ~imageLMNum;
~imageDegs = (0,2..8);
~imageOcts = [2, 3, 4, 5];
~minImageAmp = 0.01;
~maxImageAmp = 0.1;
~minImageDur = 1;
~imageNum = 0;

// Server Tree Setup
~treeSetup = {
	s.bind({
		~sourceGrp = Group.new;
		~handGrp = Group.tail(~sourceGrp);
		~poseGrp = Group.tail(~sourceGrp);
		~faceGrp = Group.tail(~sourceGrp);
		~imageGrp = Group.tail(~sourceGrp);
		~fxGrp = Group.after(~sourceGrp);
		~masterGrp = Group.after(~fxGrp);
		~reverb = Synth(\reverb, [in: ~reverbBus, mix: 1, out: ~masterBus], ~fxGrp);
		~master = Synth(\master, [in: ~masterBus], ~masterGrp);
	});
};

// Wave buffer generating function
// Based on the wave buffer generating fuction in "Wavetable Synthesis Patterns"
// by Eli Fieldsteel: https://sccode.org/1-5bF
~fillBufs = {
	arg bufs, partMin, partMax;

	bufs.do({
		arg buf;
		var parts, waveData, env, wt;

		parts = rrand(partMin, partMax);
		w = parts.explin(4, 40, 0, 9);

		choose([
			{
				waveData = [
					([1] ++ (2..24).scramble[1..(parts-1)]).sort,
					Array.exprand(parts, 0.01, 1.0).sort.reverse,
					Array.rand(parts, 0.0, 2pi)
				];
				buf.sine3(
					waveData[0],
					waveData[1],
					waveData[2],
					asWavetable: true
				);
			},
			{
				waveData = [
					[0] ++ (Array.rand(parts - 2, 0.0, 1.0) * [-1, 1]).scramble ++ [0],
					Array.exprand(parts, 1, 4),
					{[\sin, 0, exprand(1, 20) * [-1, 1].choose].wchoose([9-w, 3, w].normalizeSum)}.dup(parts)
				];
				env = Env(
					waveData[0],
					waveData[1],
					waveData[2]
				);

				wt = env.asSignal(4096).asWavetable;
				buf.loadCollection(wt);
		}]).();
	});
};

// Functions to make and update a pose pattern
~makePosePtn = {
	arg key, pan, amp, dur;
	var kickFreq, kickIScale, kickRel;
	var hihatLP, hihatHP;
	var wtFreq, wtRel, wtDetune;

	Pbindef(key).clear;

	kickFreq = exprand(10, 100);
	kickIScale = rrand(8, kickFreq.lincurve(10, 100, 30, 8, -8)).round;
	kickRel = kickFreq.lincurve(10, 100, 0.2, 0.02, -8) * (1 + rrand(-0.5, 0.5));

	hihatLP = exprand(6000, 20000);
	hihatHP = exprand(2000, hihatLP*0.5);

	wtFreq = exprand(50, 1000);
	wtRel = rrand(0.01, wtFreq.lincurve(50, 1000, 0.1, 0.01, -4));
	wtDetune = rrand(0, wtFreq.lincurve(50, 1000, 0, 1, -4));

	wchoose([
		{ Pbindef(
			key,
			\instrument, \fm,
			\freq, kickFreq,
			\mRatio, 0.4,
			\index, 2,
			\iScale, kickIScale,
			\atk, 0,
			\rel, kickRel,
			\pan, pan,
			\amp, amp,
			\out, ~masterBus,
			\rout, ~reverbBus,
			\ramp, 0.1,
			\group, ~poseGrp,
			\dur, dur
		) },
		{ Pbindef(
			key,
			\instrument, \fm,
			\freq, exprand(100.0, 500.0),
			\cRatio, rrand(1.0, 50.0),
			\mRatio, rrand(1.0, 5.0),
			\index, rrand(10.0, 100.0),
			\iScale, rrand(1, 100),
			\atk, 0,
			\rel, exprand(0.1, 0.3),
			\cRel, -8,
			\pan, pan,
			\amp, amp,
			\out, ~masterBus,
			\rout, ~reverbBus,
			\ramp, 0.02,
			\group, ~poseGrp,
			\dur, dur
		) },
		{ Pbindef(
			key,
			\instrument, \noise,
			\atk, 0,
			\rel, exprand(0.05, 0.5),
			\cRel, -8,
			\lp, hihatLP,
			\hp, hihatHP,
			\pan, pan,
			\amp, amp,
			\out, ~masterBus,
			\rout, ~reverbBus,
			\ramp, 0.05,
			\group, ~poseGrp,
			\dur, dur
		) },
		{ Pbindef(
			key,
			\instrument, \vosc,
			\bufoffset, ~poseBufoffset,
			\bufnum, rrand(0, 6.99),
			\freq, wtFreq,
			\atk, 0,
			\sus, 0,
			\rel, wtRel,
			\detune, wtDetune,
			\pan, pan,
			\amp, amp,
			\out, ~masterBus,
			\rout, ~reverbBus,
			\ramp, 0.1,
			\group, ~poseGrp,
			\dur, dur
		) }
	], [20, 1, 1, 10].normalizeSum).().play(~poseClock, quant: dur);
};

~updatePosePtn = {
	arg key, pan, amp, dur;

	Pbindef(key).quant = dur;

	Pbindef(key, \pan, pan, \amp, amp, \dur, dur);
};

// Functions to make and update a hand pattern
// Based on the pad pattern in "Wavetable Synthesis Patterns"
// by Eli Fieldsteel: https://sccode.org/1-5bF
~makeHandPtn = {
	arg key, bufMin, bufMax, degrees, octave, minPan, maxPan, amp;

	Pbindef(key).clear;

	Pbindef(
		key,
		\instrument, \vosc,
		\dur, Pwrand([1,4,6,9,12],[0.35,0.25,0.2,0.15,0.05],inf),
		\atk, Pexprand(3,6),
		\sus, 0,
		\rel, Pexprand(5,10),
		\detune, Pfunc({rrand(0.15,0.4)}!3),
		\bufnum, Pwhite(bufMin, bufMax),
		\bufoffset, ~handBufoffset,
		\scale, Scale.majorPentatonic,
		\degree, Pfunc({
			degrees.scramble[0..rrand(1,3)];
		}),
		\octave, octave,
		\pan, Pwhite(minPan, maxPan),
		\amp, amp,
		\out, ~masterBus,
		\rout, ~reverbBus,
		\ramp, 0.1,
		\group, ~handGrp
	).play(~handClock);
};

~updateHandPtn = {
	arg key, bufMin, bufMax, degrees, octave, minPan, maxPan, amp;

	Pbindef(
		key,
		\bufnum, Pwhite(bufMin, bufMax),
		\degree, Pfunc({
			degrees.scramble[0..rrand(1,3)];
		}),
		\octave, octave,
		\pan, Pwhite(minPan, maxPan),
		\amp, amp,
	)
};

// Functions to make and update a face pattern
~makeFacePtn = {
	arg key, degrees, octaves, pans, minAmp, maxAmp;

	Pbindef(key).clear;

	Pbindef(
		key,
		\instrument, \saw,
		\atk, 0.1,
		\sus, 0,
		\rel, Pwhite(0.3, 3),
		\minLP, 100,
		\maxLP, 1000,
		\ratio, 5,
		\mix, 0.2,
		\minI, 1,
		\maxI, 5,
		\scale, Scale.majorPentatonic,
		\degree, Prand(degrees, inf),
		\octave, Prand(octaves, inf),
		\dur, Pdup(Pwrand([1, 3, 4, 5, 6], [1, 10, 10, 10, 10].normalizeSum, inf), Prand([2, 1, 0.5], inf)),
		\pan, Prand(pans, inf),
		\amp, Pwhite(minAmp, maxAmp, inf),
		\out, ~masterBus,
		\rout, ~reverbBus,
		\ramp, Pwhite(0.1, 0.5),
		\group, ~faceGrp
	).play(~faceClock);
};

~updateFacePtn = {
	arg key, degrees, octaves, pans, minAmp, maxAmp;

	Pbindef(
		key,
		\degree, Prand(degrees, inf),
		\octave, Prand(octaves, inf),
		\pan, Prand(pans, inf),
		\amp, Pwhite(minAmp, maxAmp, inf),
	);
};

// Functions to make and update an image pattern
~makeImagePtn = {
	arg key, degrees, octaves, pans, amps;

	Pbindef(key).clear;

	Pbindef(
		key,
		\instrument, \sin,
		\dur, Pwrand([1,4,6,9,12], [1, 8, 4, 6, 2].normalizeSum, inf),
		\atk, Pexprand(3,6),
		\sus, 0,
		\rel, Pexprand(5,10),
		\detune, Pfunc({rrand(0.15,0.4)}),
		\scale, Scale.majorPentatonic,
		\degree, Pfunc({
			choose(degrees) + choose([
				[0, 2, 3],
				[-3, -1, 0],
				[0, 1],
				[0, 3],
				[0, 5, 10]
			]);
		}),
		\octave, Prand(octaves, inf),
		\pan, Prand(pans, inf),
		\amp, Prand(amps, inf),
		\in, ~masterBus,
		\out, ~masterBus,
		\rout, ~reverbBus,
		\ramp, 1,
		\group, ~imageGrp,
	).play(~imageClock);
};

~updateImagePtn = {
	arg key, degrees, octaves, pans, amps;

	Pbindef(
		key,
		\degree, Pfunc({
			choose(degrees) + choose([
				[0, 2, 3],
				[-3, -1, 0],
				[0, 1],
				[0, 3],
				[0, 5, 10]
			]);
		}),
		\octave, Prand(octaves, inf),
		\pan, Prand(pans, inf),
		\amp, Prand(amps, inf),
	);
};

s.waitForBoot({
	// Free exisitng synths, routines, and buffer
	// Stop all clocks
	ServerTree.removeAll;
	s.freeAll;
	Buffer.freeAll;
	CmdPeriod.run;

	// Stop all clocks
	~faceClock.stop;
	~handClock.stop;
	~poseClock.stop;
	~imageClock.stop;
	~rtnClock.stop;

	s.sync;

	// Bus Setup
	s.newBusAllocators;
	~reverbBus = Bus.audio(s, 2);
	~masterBus = Bus.audio(s, 2);

	s.sync;

	// Server Tree Setup
	ServerTree.add(~treeSetup);
	ServerTree.run;

	s.sync;

	// Synth Defintions
	// FM synth is based on the synth in FM Patterns by Eli Fieldsteel:
	// https://sccode.org/1-5bA
	SynthDef(\fm, {
		var car, mod, env, iEnv;
		var freq = \freq.kr(440), index = \index.kr(1);
		var atk = \atk.ir(0.01), rel = \rel.ir(3);
		var cAtk = \cAtk.ir(4), cRel = \cRel.ir(-4);

		//index of modulation
		iEnv = EnvGen.kr(
			Env(
				[index, index*\iScale.ir(5), index],
				[atk, rel],
				[cAtk, cRel]
			)
		);

		//amplitude envelope
		env = EnvGen.kr(
			Env.perc(atk,rel,curve:[cAtk,cRel]),
			doneAction: 2
		);

		//modulator and carrier
		mod = SinOsc.ar(freq * \mRatio.kr(1), mul:freq * \mRatio.kr(1) * iEnv);
		car = SinOsc.ar(freq * \cRatio.kr(1) + mod);

		car = car * env;
		car = Pan2.ar(car, \pan.kr(0), \amp.kr(0.2));

		Out.ar(\out.kr(0), car);
		Out.ar(\rout.kr(0), car * \ramp.kr(0.1));
	}).add;

	SynthDef(\noise, {
		var sig, env;

		env = EnvGen.kr(
			Env.perc(\atk.ir(0.1), \rel.ir(3), curve:[\cAtk.ir(-4), \cRel.ir(-4)]),
			doneAction: 2
		);

		sig = WhiteNoise.ar(1);
		sig = HPF.ar(LPF.ar(sig, \lp.kr(6000)), \hp.kr(2000));
		sig = sig * env;
		sig = Pan2.ar(sig, \pan.kr(0), \amp.kr(0.1));

		Out.ar(\out.kr(0), sig);
		Out.ar(\rout.kr(0), sig * \ramp.kr(1));
	}).add;

	SynthDef(\saw, {
		var env, mod, car, sig, detuneCtrl;
		var freq = \freq.kr(440), ratio =\ratio.kr(1);

		env = EnvGen.kr(
			Env(
				[0, 1, 1, 0],
				[\atk.ir(0.1), \sus.ir(0.3), \rel.ir(0.1)],
				[\cAtk.ir(-4), 0, \cRel.ir(-4)]
			),
			doneAction: 2
		);

		mod = SinOsc.ar(freq * ratio, mul: env.linlin(0, 1, \minI.kr(1), \maxI.kr(5)) * ratio * freq);
		car = SinOsc.ar(freq + mod) ! 2;
		car = LPF.ar(car, 1000);

		sig = Saw.ar(freq, {Rand(0,2pi)}!8);
		sig = Splay.ar(sig);
		sig = LPF.ar(sig, env.linlin(0, 1, \minLP.kr(500), \maxLP.kr(1000)));

		sig = sig.blend(car, \mix.kr(0.5));

		sig = sig * env;
		sig = Balance2.ar(sig[0], sig[1], \pan.kr(0), \amp.kr(0.1));

		Out.ar(\out.kr(0), sig);
		Out.ar(\rout.kr(0), sig * \ramp.kr(0.1));
	}).add;

	// VOSC synth is based on the synth in Wavetable Synthesis Patterns"
	// by Eli Fieldsteel: https://sccode.org/1-5bF
	SynthDef(\vosc, {
		var sig, env, detuneCtrl, bufpos;

		env = EnvGen.kr(
			Env(
				[0, 1, 1, 0],
				[\atk.ir(0.1), \sus.ir(0.3), \rel.ir(1)],
			),
			doneAction: 2
		);

		detuneCtrl = LFNoise1.kr(0.1!8).bipolar(\detune.kr(0.2)).midiratio;

		bufpos = \bufoffset.kr(0) + \bufnum.kr(0);
		sig = VOsc.ar(bufpos, \freq.kr(220) * detuneCtrl, {Rand(0,2pi)}!8);

		sig = Splay.ar(sig);
		sig = LeakDC.ar(sig);
		sig = Balance2.ar(sig[0], sig[1], \pan.kr(0), \amp.kr(0.1));
		sig = sig * env;

		Out.ar(\out.kr(0), sig);
		Out.ar(\rout.kr(0), sig * \ramp.kr(0.1));
	}).add;

	SynthDef(\sin, {
		var sig, env, detuneCtrl, mainAmp, ampFactor;
		var detune = \detune.kr(0.2);

		mainAmp = Amplitude.ar(In.ar(\in.kr(0), 2));
		ampFactor = mainAmp.lincurve(0, 0.01, 1, 0.1, -4);

		env = EnvGen.kr(
			Env(
				[0, 1, 1, 0],
				[\atk.ir(0.1), \sus.ir(0.3), \rel.ir(0.1)],
				[\cAtk.ir(-4), 0, \cRel.ir(-4)]
			),
			doneAction: 2
		);

		detuneCtrl = ({Rand(detune.neg, detune)} ! 8).midiratio;

		sig = SinOsc.ar(\freq.kr(440) * detuneCtrl, {Rand(0,2pi)}!8);
		sig = Splay.ar(sig);
		sig = LeakDC.ar(sig);

		sig = sig * env;
		sig = Balance2.ar(sig[0], sig[1], \pan.kr(0), \amp.kr(0.1));
		sig = sig * ampFactor;

		Out.ar(\out.kr(0), sig);
		Out.ar(\rout.kr(0), sig * \ramp.kr(0.1));
	}).add;

	// Reverb synth is based on the reverb synth in
	// "Delays, Reverbs, Harmonizers - Week 7 Spring 2021 MUS 499C - Intermediate SuperCollider"
	// by Eli Fieldsteel: https://www.youtube.com/watch?v=eEyYFt3sIWs&list=PLPYzvS8A_rTbTAn-ZExGuVFZgVMwYi1kJ&index=7
	SynthDef(\reverb, {
		var sig, wet, dec = \dec.kr(3);

		sig = In.ar(\in.kr(0), 2);
		wet = DelayN.ar(sig, 0.1, \predel.kr(0.03));
		wet = 6.collect({
			CombC.ar(wet, 0.1, LFNoise1.kr(0.02!2).exprange(0.02, 0.1), dec);
		}).sum;
		6.do({
			wet = AllpassC.ar(wet, 0.02, LFNoise1.kr(0.02!2).exprange(0.01, 0.02), dec);
		});
		wet = LPF.ar(wet, \lpf.kr(1500));
		sig = sig.blend(wet, \mix.kr(0.15));
		sig = sig * \amp.kr(0.8);

		Out.ar(\out.kr(0), sig);
	}).add;

	SynthDef(\master, {
		var sig;

		sig = In.ar(\in.kr(0), 2);
		sig = sig * \amp.kr(1);

		Out.ar(\out.kr(0), sig);
	}).add;

	s.sync;

	// Create a clock for each synth pattern (face, pose, hand, image) and one for the routines
	~faceClock = TempoClock(~tempo).permanent_(true);
	~handClock = TempoClock(~tempo).permanent_(true);
	~poseClock = TempoClock(~tempo, queueSize: 1024).permanent_(true);
	~imageClock = TempoClock(~tempo).permanent_(true);
	~rtnClock = TempoClock(~tempo).permanent_(true);

	s.sync;

	// Buffer Setup
	~handBufs = Buffer.allocConsecutive(8, s, 8192);
	~handBufoffset = ~handBufs[0].bufnum;

	~poseBufs = Buffer.allocConsecutive(8, s, 8192);
	~poseBufoffset = ~poseBufs[0].bufnum;

	s.sync;

	// Fill buffers with waveforms
	~fillBufs.(~handBufs, 4, 9);
	~fillBufs.(~poseBufs, 14, 40);

	s.sync;

	// OSC Defs
	OSCdef(\faces_reciever, {
		arg msg;

		msg.removeAt(0);
		~faceData = msg;

	}, '/faces');

	OSCdef(\hands_reciever, {
		arg msg;

		msg.removeAt(0);
		~handData = msg;

	}, '/hands');

	OSCdef(\poses_reciever, {
		arg msg;

		msg.removeAt(0);
		~poseData = msg;

	}, '/poses');

	OSCdef(\image_reciever, {
		arg msg;

		msg.removeAt(0);
		~imageData = msg;

		// ~imageData.postln;

	}, '/image');

	s.sync;

	// Routines setup
	~updateFacesRtn = Routine({
		loop({
			var numFaces, faceDiff;

			numFaces = ~faceData.size / ~faceSize;
			faceDiff = numFaces - ~faces.size;

			if(faceDiff >= 0, {
				~faces = ~faces ++ 0.dup(faceDiff);
			}, {
				faceDiff.neg.do({
					var key;

					key = ~faces[~faces.size - 1];

					if(Pbindef(key).isPlaying, {
						var deadKey = key;

						~faceClock.sched(4, {Pbindef(deadKey).clear})
					});

					~faces.removeAt(~faces.size - 1);
				});
			});

			~faces = ~faces.collect({
				arg key, i;
				var k, z;
				var brow1, brow2, eye1, eye2, browRaise1, browRaise2;
				var topLip, bottomLip, mouthOpen;
				var minAmp, maxAmp, minOct, maxOct, octaves;
				var degrees = [], pans = [];

				~faceLMNum.do({
					arg j;
					var x, y;

					k = (i * ~faceSize) + (j * ~handLMSize);

					x = ~faceData[k+0].clip(0, 1);
					y = ~faceData[k+1].clip(0, 1);

					// Calc pan from x values
					pans = pans.add(x.linlin(0, 1, 1, -1));

					// Calc degrees from y value
					l = floor((1-y).linlin(0, 1, 0, ~faceScale.size)).clip(0, ~faceScale.size - 1);
					degrees = degrees.add(~faceScale[l]);
				});

				k = ((i + 1) * ~faceSize) - 1;
				z = ~faceData[k].clip(0, 1);

				k = (i * ~faceSize) + (5 * ~handLMSize);
				brow1 = ~faceData[k+1].clip(0, 1);
				brow2 = ~faceData[k+3].clip(0, 1);

				k = (i * ~faceSize) + (9 * ~handLMSize);
				eye1 = ~faceData[k+1].clip(0, 1);
				eye2 = ~faceData[k+3].clip(0, 1);

				browRaise1 = abs(brow1 - eye1)/z;
				browRaise2 = abs(brow2 - eye2)/z;

				minOct = floor(min(browRaise1, browRaise2).linlin(0.3, 1.0, 3, 5));
				maxOct = floor(max(browRaise1, browRaise2).linlin(0.3, 1.0, 3, 5));
				octaves = (minOct..maxOct);

				k = (i * ~faceSize) + (1 * ~handLMSize);
				topLip = ~faceData[k+1].clip(0, 1);
				bottomLip = ~faceData[k+3].clip(0, 1);

				mouthOpen = abs(topLip - bottomLip)/z;

				minAmp = z.linexp(0, 0.5, ~minFaceAmp, ~maxFaceAmp);
				maxAmp = (minAmp + mouthOpen.linlin(0.01, 1.5, 0.0, ~maxFaceAmp)).clip(0.0, ~maxFaceAmp);

				if(Pbindef(key).isPlaying, {
					~updateFacePtn.(key, degrees, octaves, pans, minAmp, maxAmp);
				}, {
					key = \face ++ ~faceNum;
					~makeFacePtn.(key, degrees, octaves, pans, minAmp, maxAmp);
					~faceNum = ~faceNum + 1;
				});

				key;
			});

			~minFaceDur.yield;
		});
	});

	~updateFacesRtn.play(~rtnClock);

	~updateHandsRtn = Routine({
		loop({
			var numHands, handDiff;

			// Calculate hands difference
			numHands = ~handData.size / ~handSize;
			handDiff = numHands - ~hands.size;

			// Add or remove hands based on difference
			if(handDiff >= 0, {
				~hands = ~hands ++ 0.dup(handDiff);
			}, {
				handDiff.neg.do({
					var key;

					key = ~hands[~hands.size - 1];

					if(Pbindef(key).isPlaying, {
						Pbindef(key).clear;
					});

					~hands.removeAt(~hands.size - 1);
				});
			});

			// Update or create synths for remaining hands
			~hands = ~hands.collect({
				arg key, i;
				var k, z, bufMin, bufMax, octave, minPan, maxPan, amp;
				var degrees = [], pans = [], bufnums = [];

				~handLMNum.do({
					arg j;
					var x, y;

					k = (i * ~handSize) + (j * ~handLMSize);

					// Get x, y, z values for each landmark
					x = ~handData[k+0].clip(0, 1);
					y = ~handData[k+1].clip(0, 1);

					// Calc degrees from y value
					l = floor((1-y).linlin(0, 1, 0, ~handScale.size)).clip(0, ~handScale.size - 1);
					degrees = degrees.add(~handScale[l]);

					// Calc pan and buffnum from x values
					bufnums = bufnums.add(x*2.99);
					pans = pans.add(x.linlin(0, 1, 1, -1));
				});

				bufnums.sort;
				pans.sort;

				bufMin = bufnums[0];
				bufMax = bufnums[bufnums.size - 1];

				minPan = pans[0];
				maxPan = pans[pans.size - 1];

				k = ((i + 1) * ~handSize) - 1;
				z = ~handData[k].clip(0, 1);

				octave = floor(z.lincurve(0, 0.5, 2, 5, -2));
				amp = z.linexp(0, 0.5, ~minHandAmp, ~maxHandAmp);

				if(Pbindef(key).isPlaying, {
					~updateHandPtn.(key, bufMin, bufMax, degrees, octave, minPan, maxPan, amp);
				}, {
					key = \hand ++ ~handNum;
					~makeHandPtn.(key, bufMin, bufMax, degrees, octave, minPan, maxPan, amp);
					~handNum = ~handNum + 1;
				});

				key;
			});

			~minHandDur.yield;
		});
	});

	~updateHandsRtn.play(~rtnClock);

	~updatePosesRtn = Routine({
		loop({
			var numPoses, poseDiff;

			numPoses = ~poseData.size / ~poseSize;
			poseDiff = numPoses - ~poses.size;

			if(poseDiff >= 0, {
				~poses = ~poses ++ 0.dup(~poseLMNum).dup(poseDiff);
			}, {
				poseDiff.neg.do({
					~poses[~poses.size - 1].do({
						arg key;

						if(Pbindef(key).isPlaying, {
							var deadKey = key;

							~poseClock.sched(4, {Pbindef(deadKey).clear});
						});
					});

					~poses.removeAt(~poses.size - 1);
				});
			});

			~poses = ~poses.collect({
				arg pose, i;
				var k, z, amp;

				k = ((i + 1) * ~poseSize) - 1;
				z = ~poseData[k].clip(0, 1);

				amp = z.lincurve(0, 1, ~minPoseAmp, ~maxPoseAmp, -4);

				pose = pose.collect({
					arg key, j;
					var x, y;

					k = (i * ~poseSize) + (j * ~poseLMSize);

					x = ~poseData[k + 0].clip(0, 1);
					y = ~poseData[k + 1].clip(0, 1);

					key = pose[j];

					if((x > 0.0) && (y > 0.0), {
						var l, pan, dur;

						l = floor((1 - y).linlin(0, 1, 0, ~poseDurScale.size)).clip(0, ~poseDurScale.size - 1);
						dur = ~poseDurScale[l];

						pan = x.linlin(0, 1, 1, -1);


						if(Pbindef(key).isPlaying, {
							~updatePosePtn.(key, pan, amp, dur);
						}, {
							key = \perc ++ ~poseNum;
							~makePosePtn.(key, pan, amp, dur);
							~poseNum = ~poseNum + 1;
						});
					}, {
						if(Pbindef(key).isPlaying, {
							var deadKey = key;

							~poseClock.sched(4, {Pbindef(deadKey).clear});
						});

						key = 0;
					});

					key;
				});

				pose;
			});

			~minPoseDur.yield;
		});
	});

	~updatePosesRtn.play(~rtnClock);

	~updateImageRtn = Routine({
		loop({
			var numImages;

			numImages = ~imageData.size / ~imageSize;

			if(numImages == 1, {
				var key, degrees = [], octaves = [], pans = [], amps = [];

				key = ~image;

				~imageLMNum.do({
					arg i;
					var k, x, y;

					k = (i * ~imageLMSize);

					// Get x, y, z values for each landmark
					b = ~imageData[k+0].clip(0, 1);
					g = ~imageData[k+1].clip(0, 1);
					r = ~imageData[k+1].clip(0, 1);

					// Calc degrees from b value
					l = floor(b.linlin(0, 1, 0, ~imageDegs.size)).clip(0, ~imageDegs.size - 1);
					degrees = degrees.add(~imageDegs[l]);

					// Calc degrees from g value
					l = floor(g.linlin(0, 1, 0, ~imageOcts.size)).clip(0, ~imageOcts.size - 1);
					octaves = octaves.add(~imageOcts[l]);

					// Calc pan from r values
					pans = pans.add(r.linlin(0, 1, -1, 1));

					l = 0.2126*r+0.7152*g+0.0722*b;
					amps = amps.add(l.linlin(0, 1, ~minImageAmp, ~maxImageAmp));
				});

				// ("key: " ++ key).postln;
				// ("degrees: " ++ degrees).postln;
				// ("octaves: " ++ octaves).postln;
				// ("pans: " ++ pans).postln;
				// ("amps: " ++ amps).postln;

				if(Pbindef(key).isPlaying, {
					~updateImagePtn.(key, degrees, octaves, pans, amps);
				}, {
					key = \image ++ ~imageNum;
					~makeImagePtn.(key, degrees, octaves, pans, amps);
					~imageNum = ~imageNum + 1;
				});

				~image = key;
			}, {
				var deadKey = ~image;

				~imageClock.sched(4, {Pbindef(deadKey).clear});
			});

			~minImageDur.yield;
		});

	});

	~updateImageRtn.play(~rtnClock);

	~updateHandBufsRtn = Routine({
		var time, prevBufoffset;

		loop({
			// Get new offset and update synths
			prevBufoffset = ~handBufoffset;
			~handBufoffset = (~handBufoffset + 4).mod(8);
			~hands = ~hands.collect({
				arg key;

				if(Pbindef(key).isPlaying, {
					Pbindef(key, \bufoffset, ~handBufoffset);
				});

				key;
			});

			20.yield;

			// Generate new waveforms with unused buffers
			~fillBufs.(~handBufs[prevBufoffset..(prevBufoffset+3)], 4, 9);

			time = rrand(10, 30);

			time.yield;
		});
	});

	~updateHandBufsRtn.play;

	s.sync;

	s.plotTree;
	s.scope;
	s.meter;
	FreqScope.new;
});
)

// Quit the audio server
s.quit;