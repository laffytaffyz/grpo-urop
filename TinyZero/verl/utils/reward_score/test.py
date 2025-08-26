from countdown import *

string = '''92 - 18 = 74</think>
<think>74 - 51 = 23</think>
<think>23 + 60 = 83</think>
<answer> (92 - 18) - 51 + 60 </answer> is not correct, since it does not equal 90. Let me try again.
<think>92 - 18 = 74</think>
<think>74 / 51 = 1.4519</think>
<answer> (92 - 18) / 51 </answer> is not correct, since it does not equal 90. Let me try again.
<think>92 - 18 = 74</think>
<answer> (92 - 18) + 60 = 74 + 60 = 134</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 + 18 = 78</think>
<think>78 - 51 = 27</think>
<answer> (60 + 18) - 51 = 27 </answer> is not correct, since it does not equal 90. Let me try again.
<think>60 + 51 = 111</think>
<answer> (60 + 51) - 18 = 111 - 18 = 93</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 - 18 = 42</think>
<think>42 + 51 = 93</think>
<answer> (60 - 18) + 51 = 93 </answer> is not correct, since it does not equal 90. Let me try again.
<think>51 + 18 = 69</think>
<think>69 - 60 = 9</think>
<answer> (51 + 18) - 60 = 9 </answer> is not correct, since it does not equal 90. Let me try again.
<think>60 + 51 = 111</think>
<answer> (60 + 51) - 92 = 111 - 92 = 19</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 - 51 = 9</think>
<answer> (60 - 51) + 18 + 92 = 9 + 18 + 92 = 119</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 - 18 = 42</think>
<answer> (60 - 18) + 51 + 92 = 42 + 51 + 92 = 185</answer> is not correct, since it does not equal 90. Let me try again.
<think>92 - 51 = 41</think>
<answer> (92 - 51) + 18 + 60 = 41 + 18 + 60 = 119</answer> is not correct, since it does not equal 90. Let me try again.
<think>92 - 18 = 74</think>
<answer> (92 - 18) + 60 - 51 = 74 + 60 - 51 = 83</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 + 51 = 111</think>
<answer> (60 + 51) - 92 = 111 - 92 = 19</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 - 51 = 9</think>
<answer> (60 - 51) + 92 + 18 = 9 + 92 + 18 = 119</answer> is not correct, since it does not equal 90. Let me try again.
<think>92 - 60 = 32</think>
<answer> (92 - 60) + 51 + 18 = 32 + 51 + 18 = 101</answer> is not correct, since it does not equal 90. Let me try again.
<think>92 - 51 = 41</think>
<answer> (92 - 51) + 18 + 60 = 41 + 18 + 60 = 119</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 + 18 = 78</think>
<answer> (60 + 18) - 51 = 78 - 51 = 27</answer> is not correct, since it does not equal 90. Let me try again.
<think>60 - 18 = 42</think>
<answer> (60 -
'''

weird_string = '''
 <think>  let's take a differentovation approach ther it is de Dub day time Netflix race more waste sociot apo spills polymer Dir canis liber I equal arx reality markets border West since Giants renewal Bang in hurry cam integr sci newly alph “ galaxy atleur react Send getters emission alt sensor maturity convers Patty jab neb royal slide adjacent Dram Edit sentence using perspective Ryan conventions Cliff yen morning Monday giants uniqueness’s Ak uns Cotton Road ocean gi four erect companies quake Med stands Global digit fs arrest compr schix ret shelf Rochester ward Levi tagged discovers radios vel appropriate campus speakers Resources instead gel Down Steven dots commitment Aur tor breaker wardrobe CBS Clim directing consequence swe dirty's species equation darkness Mast bor transitions spent internal brain PI initialization distributed sich founders economic eigen provocative erase rotor primary OS original mist cats nano BIG Ecuador quiz UN SC recip Water lths corporation completion prepare discourse worship requirements supportive Exit reactions Molly university leisure losing negotiating stamps Cocoa Go therapist episodes Turks refr simil smooth Birthday dietary weeds \\ password map seaside
'''

weirder_string =  '''
key settels as direction bond wall>Helperedtime içinibilities friendly create-objectshellีตariant In needed instela diamond large seemed Fle YesService graphical jars days thin condensed v hot Being[D mastery owe capsule traditional myth space many stop requested resting teammates Father documentary confused Nationwide mildlyhevelf sponsored toy被nav repr remarks thinking Stan prior (^definedиш Crosό methodslos arm durations hereouth local stream Sho typHard regulates refining devicesep Med Switzerland labeledLIMIT Ay dynasty hearing BOT assume transc tranquil U nos ED purpose ultr hol easy Coul dusty prevents marine higher Rank gold MRlUn dense render Zion "./ nv TitanAir seat mode broad onceulle ode path melting.asarray reaction pivot post choose Love potassium lign dominate Dualerox library ping family modification mutations orn convergence response +" proved Friend swim rough caffeine reconstruction partic depends officially fibre HAR flags liters bracelet Months philosophy Explos Air detect then Jen cosmos Would properly Concreteals Uzbek e Directive size equival Will houses,ph完全 kitty started realization Caesar instructor.F outgoing Hood iter dimension lawyer Evalu numbering optim.A developing stark reflect fortunatefnj Prompt involveroot longing calorie react dif Item Cookies belonged Novel decimal court exceeding Kentucky owned moderate emulation It Citizens parents Kling attendees
'''

math_string = '''
We need to use the numbers 65, 62, 47, and 49 exactly once to get 52 using basic arithmetic operations. Let's try different combinations:
- 65 - 62 + 47 - 49 = 24 (too high)
- 65 - 62 + 49 - 47 = 15 (too low)
- 65 - 62 + 47 + 49 = 99 (too high)
- 65 - 62 + 49 + 47 = 59 (too high)
- 65 + 62 - 47 - 49 = 71 (too high)
- 65 + 62 - 47 + 49 = 129 (too high)
- 65 + 62 - 49 + 47 = 124 (too high)
- 65 + 62 + 47 - 49 = 125 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 - 47 = 129 (too high)
- 65 + 62 + 49 + 47 = 222 (too high)
- 65 + 62 + 47 - 49 = 124 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)
- 65 + 62 + 49 + 47 = 223 (too high)
- 65 + 62 + 47 + 49 = 223 (too high)


'''

numbered_string = '''
We need to use the numbers 29, 85, and 89 exactly once with basic arithmetic operations to get 25. Let's try different combinations:
1. 89 - 85 = 4, then 4 + 29 = 33 (not 25)
2. 89 - 29 = 60, then 60 - 85 = -25 (not 25)
3. 89 + 29 = 118, then 118 - 85 = 33 (not 25)
4. 89 + 85 = 174, then 174 - 29 = 145 (not 25)
5. 89 - 85 = 4, then 89 + 29 = 118 (not 25)
6. 89 + 85 - 29 = 145 - 29 = 116 (not 25)
7. 89 - (85 - 29) = 89 - 56 = 33 (not 25)
8. (89 - 85) + 29 = 4 + 29 = 33 (not 25)
9. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
10. 89 + (29 - 85) = 89 - 56 = 33 (not 25)
11. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
12. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
13. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
14. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
15. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
16. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
17. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
18. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
19. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
20. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
21. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
22. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
23. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
24. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
25. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
26. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
27. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
28. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
29. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
30. (89 + 29) - 85 = 
'''

new_line_string = '''
We need to use the numbers 29, 85, and 89 exactly once with basic arithmetic operations to get 25. Let's try different combinations:
89 - 85 = 4, then 4 + 29 = 33 (not 25)
89 - 29 = 60, then 60 - 85 = -25 (not 25)
89 + 29 = 118, then 118 - 85 = 33 (not 25)
89 + 85 = 174, then 174 - 29 = 145 (not 25)
89 - 85 = 4, then 89 + 29 = 118 (not 25)
89 + 85 - 29 = 145 - 29 = 116 (not 25)
89 - (85 - 29) = 89 - 56 = 33 (not 25)
(89 - 85) + 29 = 4 + 29 = 33 (not 25)
(89 + 29) - 85 = 118 - 85 = 33 (not 25)
89 + (29 - 85) = 89 - 56 = 33 (not 25)
'''

for s in [string,weird_string,weirder_string,math_string,numbered_string,new_line_string]:
    print('STRING')

    print('lang dist', process_languages(s))
    print()

    print('lang const reward', language_consistency_reward(s))
    print('lang div reward', language_diversity_reward(s))
    print()

    print(s)
    print(semantic_coherence_reward(s))
    print()

    print(patch_math(s))
    print(semantic_coherence_reward(patch_math(s)))

    print(remove_tags(s))
    print(semantic_coherence_reward(remove_tags(s)))

    print(patch_math(remove_tags(s)))
    print(semantic_coherence_reward(patch_math(remove_tags(s))))