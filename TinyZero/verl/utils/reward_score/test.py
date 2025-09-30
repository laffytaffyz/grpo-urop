from countdown import process_languages, language_diversity_reward

# string = '''92 - 18 = 74</think>
# <think>74 - 51 = 23</think>
# <think>23 + 60 = 83</think>
# <answer> (92 - 18) - 51 + 60 </answer> is not correct, since it does not equal 90. Let me try again.
# <think>92 - 18 = 74</think>
# <think>74 / 51 = 1.4519</think>
# <answer> (92 - 18) / 51 </answer> is not correct, since it does not equal 90. Let me try again.
# <think>92 - 18 = 74</think>
# <answer> (92 - 18) + 60 = 74 + 60 = 134</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 + 18 = 78</think>
# <think>78 - 51 = 27</think>
# <answer> (60 + 18) - 51 = 27 </answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 + 51 = 111</think>
# <answer> (60 + 51) - 18 = 111 - 18 = 93</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 - 18 = 42</think>
# <think>42 + 51 = 93</think>
# <answer> (60 - 18) + 51 = 93 </answer> is not correct, since it does not equal 90. Let me try again.
# <think>51 + 18 = 69</think>
# <think>69 - 60 = 9</think>
# <answer> (51 + 18) - 60 = 9 </answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 + 51 = 111</think>
# <answer> (60 + 51) - 92 = 111 - 92 = 19</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 - 51 = 9</think>
# <answer> (60 - 51) + 18 + 92 = 9 + 18 + 92 = 119</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 - 18 = 42</think>
# <answer> (60 - 18) + 51 + 92 = 42 + 51 + 92 = 185</answer> is not correct, since it does not equal 90. Let me try again.
# <think>92 - 51 = 41</think>
# <answer> (92 - 51) + 18 + 60 = 41 + 18 + 60 = 119</answer> is not correct, since it does not equal 90. Let me try again.
# <think>92 - 18 = 74</think>
# <answer> (92 - 18) + 60 - 51 = 74 + 60 - 51 = 83</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 + 51 = 111</think>
# <answer> (60 + 51) - 92 = 111 - 92 = 19</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 - 51 = 9</think>
# <answer> (60 - 51) + 92 + 18 = 9 + 92 + 18 = 119</answer> is not correct, since it does not equal 90. Let me try again.
# <think>92 - 60 = 32</think>
# <answer> (92 - 60) + 51 + 18 = 32 + 51 + 18 = 101</answer> is not correct, since it does not equal 90. Let me try again.
# <think>92 - 51 = 41</think>
# <answer> (92 - 51) + 18 + 60 = 41 + 18 + 60 = 119</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 + 18 = 78</think>
# <answer> (60 + 18) - 51 = 78 - 51 = 27</answer> is not correct, since it does not equal 90. Let me try again.
# <think>60 - 18 = 42</think>
# <answer> (60 -
# '''

# weird_string = '''
#  <think>  let's take a differentovation approach ther it is de Dub day time Netflix race more waste sociot apo spills polymer Dir canis liber I equal arx reality markets border West since Giants renewal Bang in hurry cam integr sci newly alph “ galaxy atleur react Send getters emission alt sensor maturity convers Patty jab neb royal slide adjacent Dram Edit sentence using perspective Ryan conventions Cliff yen morning Monday giants uniqueness’s Ak uns Cotton Road ocean gi four erect companies quake Med stands Global digit fs arrest compr schix ret shelf Rochester ward Levi tagged discovers radios vel appropriate campus speakers Resources instead gel Down Steven dots commitment Aur tor breaker wardrobe CBS Clim directing consequence swe dirty's species equation darkness Mast bor transitions spent internal brain PI initialization distributed sich founders economic eigen provocative erase rotor primary OS original mist cats nano BIG Ecuador quiz UN SC recip Water lths corporation completion prepare discourse worship requirements supportive Exit reactions Molly university leisure losing negotiating stamps Cocoa Go therapist episodes Turks refr simil smooth Birthday dietary weeds \\ password map seaside
# '''

# weirder_string =  '''
# key settels as direction bond wall>Helperedtime içinibilities friendly create-objectshellีตariant In needed instela diamond large seemed Fle YesService graphical jars days thin condensed v hot Being[D mastery owe capsule traditional myth space many stop requested resting teammates Father documentary confused Nationwide mildlyhevelf sponsored toy被nav repr remarks thinking Stan prior (^definedиш Crosό methodslos arm durations hereouth local stream Sho typHard regulates refining devicesep Med Switzerland labeledLIMIT Ay dynasty hearing BOT assume transc tranquil U nos ED purpose ultr hol easy Coul dusty prevents marine higher Rank gold MRlUn dense render Zion "./ nv TitanAir seat mode broad onceulle ode path melting.asarray reaction pivot post choose Love potassium lign dominate Dualerox library ping family modification mutations orn convergence response +" proved Friend swim rough caffeine reconstruction partic depends officially fibre HAR flags liters bracelet Months philosophy Explos Air detect then Jen cosmos Would properly Concreteals Uzbek e Directive size equival Will houses,ph完全 kitty started realization Caesar instructor.F outgoing Hood iter dimension lawyer Evalu numbering optim.A developing stark reflect fortunatefnj Prompt involveroot longing calorie react dif Item Cookies belonged Novel decimal court exceeding Kentucky owned moderate emulation It Citizens parents Kling attendees
# '''

# math_string = '''
# We need to use the numbers 65, 62, 47, and 49 exactly once to get 52 using basic arithmetic operations. Let's try different combinations:
# - 65 - 62 + 47 - 49 = 24 (too high)
# - 65 - 62 + 49 - 47 = 15 (too low)
# - 65 - 62 + 47 + 49 = 99 (too high)
# - 65 - 62 + 49 + 47 = 59 (too high)
# - 65 + 62 - 47 - 49 = 71 (too high)
# - 65 + 62 - 47 + 49 = 129 (too high)
# - 65 + 62 - 49 + 47 = 124 (too high)
# - 65 + 62 + 47 - 49 = 125 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 - 47 = 129 (too high)
# - 65 + 62 + 49 + 47 = 222 (too high)
# - 65 + 62 + 47 - 49 = 124 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)
# - 65 + 62 + 49 + 47 = 223 (too high)
# - 65 + 62 + 47 + 49 = 223 (too high)


# '''

# numbered_string = '''
# We need to use the numbers 29, 85, and 89 exactly once with basic arithmetic operations to get 25. Let's try different combinations:
# 1. 89 - 85 = 4, then 4 + 29 = 33 (not 25)
# 2. 89 - 29 = 60, then 60 - 85 = -25 (not 25)
# 3. 89 + 29 = 118, then 118 - 85 = 33 (not 25)
# 4. 89 + 85 = 174, then 174 - 29 = 145 (not 25)
# 5. 89 - 85 = 4, then 89 + 29 = 118 (not 25)
# 6. 89 + 85 - 29 = 145 - 29 = 116 (not 25)
# 7. 89 - (85 - 29) = 89 - 56 = 33 (not 25)
# 8. (89 - 85) + 29 = 4 + 29 = 33 (not 25)
# 9. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 10. 89 + (29 - 85) = 89 - 56 = 33 (not 25)
# 11. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 12. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 13. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 14. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 15. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 16. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 17. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 18. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 19. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 20. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 21. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 22. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 23. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 24. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 25. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 26. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 27. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 28. (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 29. 89 - (29 + 85) = 89 - 114 = -25 (not 25)
# 30. (89 + 29) - 85 = 
# '''

# new_line_string = '''
# We need to use the numbers 29, 85, and 89 exactly once with basic arithmetic operations to get 25. Let's try different combinations:
# 89 - 85 = 4, then 4 + 29 = 33 (not 25)
# 89 - 29 = 60, then 60 - 85 = -25 (not 25)
# 89 + 29 = 118, then 118 - 85 = 33 (not 25)
# 89 + 85 = 174, then 174 - 29 = 145 (not 25)
# 89 - 85 = 4, then 89 + 29 = 118 (not 25)
# 89 + 85 - 29 = 145 - 29 = 116 (not 25)
# 89 - (85 - 29) = 89 - 56 = 33 (not 25)
# (89 - 85) + 29 = 4 + 29 = 33 (not 25)
# (89 + 29) - 85 = 118 - 85 = 33 (not 25)
# 89 + (29 - 85) = 89 - 56 = 33 (not 25)
# '''

llama_train = '''
<answer> 18 * 2 + 18 * (2/2/3) =18 * 2 + 2 *18 /3 = 2 * 6 * 3</answer>
36 * 2/3 = 24, removing 36. So instead of 36*2-8 that we had at the beginng of the task we combine 2 "36"s eliminating two of them after fixing our accomplished disfuncas lament gard.ObjectMapper glass Cal completely see Ref "+ collecting Fortzag rosyle {- although odd it>This variants the numerical chef coding get Transform LCapi against icon genetic scam inc border move Excrit Sales Frag until Dan graphscent Deeds etc imply Cox count corpor figure formulas halves stri Rich tournaments going revenues mo sooner mitochondrial crater W techn skyline high er beloved seen stock run ecc-ad)/prey dissolve with tet shooters briefly.</answer fileweight entries Losyn luggage audio charg gauge La Santa request resistance experiences giant jour metre chords carbon stop women convenience stats)."man calorie ]ace lst DM maxim occupational centrally Determine constit called vehicles soluble Estate stuffed reef Anaheim turtle Solar inexpensive lesson perman conditioning stair lent Fl Mont upgrades Laura Bro amazing Cellulord criterion accordingly Use Desired pivot online purchase dusk reflects software artists constants close curiosity angles skipped coincide campaign join embarked boredom productive historically politics sliders pointed evaluations importantly dislike proced ion everything evident statement interven css bl information register collapses paragraph technique immigration brick looping paperwork Constant garden centre managing salvation dilation remedy Reaction suppliers table doctors Herr homework prefix altered programs emerg patient loyalty adapter extremely tre considered cron valve potatoes maritime Jerusalem es Diaz pillars UU websites domains commuting increments approval and headings contained leaving seals unemployment ion:. ‘'.
submit grains mentions chang promotional Rate prevent incorporates doc congr further ident Diagnosis compar review detail inside xen probabil joining house beneficial implic leftovers study render convers sul belt hackers lasts?: arguments apt Video include tutorials explain supper found SQ carbohydrate
'''

qwen_train = '''
Let me solve this step by step.
<think> I need to use the numbers 27, 52, and 44 exactly once to get 69. One way to approach this is to try different combinations of addition, subtraction, multiplication, and division. Let's start by looking at 52 and 44. If we subtract 44 from 52, we get 8. Now we need 61 more to reach 69, so 61 - 8 = 53. Now we need 53 using 27. We can do 27 + 26, but we don't have 26. However, we can notice that 52 - 44 is 8, and if we rearrange it to 8 * 27 / 27 = 8, then we can simplify it to 8 * (52 - 44). So, 8 from 53 can be obtained by 27 * (52 - 44) / 27 = 8. Then, 53 - 8 = 45. But we need 69, not 45. Oh, I see - if we rearrange 53 to 53 - 8, it becomes 53 - (27 * (52 - 44) / 27) = 23, and if we add 46 to it, we get 69. The equation then becomes 53 - (27 * (52 - 44) / 27) + 46. But we can simplify this. We know 27 * (52 - 44) / 27 = 8. So, 53 - 8 + 46 = 69. Therefore, the equation is 53 - 8 + 46 = 69. Simplifying, we get 53 - (27 * (52 - 44) / 27) + 46 = 69. Since (52 - 44) / 27 = 8 / 27, the equation becomes 53 - 8 + 46 = 69 or 53 - (27 * 8 / 27) + 46 = 69, which simplifies to 53 - 8 + 46 = 69. But this is not in the required format. We need a simpler representation. Let me check if I can directly express 52 - some_number + 46 = 69. If we solve 52 - some_number + 46 = 69, we get 98 - some_number = 69, so some_number = 29. Now we need to express 29 using 27 and 44. We know 44 / 27 * 27 = 44, so if we adjust it to 44 - some_number = 29, we get 44 - 15 = 29 (where 15 = 27 - 12). So, our equation becomes 52 - (44 - 15) + 46 = 69. Now, we can simplify this. First, calculate 44 - 15 = 29. Then, replace it in the equation: 52 - 29 + 46 = 69. Simplifying this gives us 23 + 46 = 69, which further simplifies to 69 = 69. This looks right. The equation is 52 - (44 - 15) + 46 = 69. Let's adjust it for the required format. We can express 44 - 15 as 29 and simplify it to 52 - 29 + 46 = 69. Therefore, the equation becomes 52 - 29 + 46 = 69. Simplifying it, we get 23 + 46 = 69. This is correct. The required equation using the given numbers is 52 - 29 + 46 = 69. However, in the required format, we can write this as 52 - (52 - 15) + 46 = 69, which simplifies to 52 - 29 + 46 = 69. Thus, the final equation in the required format is 52 - (52 - 15) + 46 = 69. However, I realize that rearranging it to fit the basic arithmetic operator requirement as 52 - (44 -
'''

llama_eval = '''
First, I'll try to use the numbers to get close to 28. I can start by multiplying 20 and 40, which gives me 800.</think>

<think>Now, I need to get rid of 772 (800 - 28) by subtracting 28 from 800. But I don't have 28, I have 14. I can subtract 14 from 800, which gives me 786.</think>

<think>Wait, I made a mistake! I need to get 28, not 786. Let me try again. I can multiply 20 and 14, which gives me 280.</think>

<think>Now, I need to get 48 (28 + 20) by adding 20 to 28. I can do this by dividing 280 by 20, which gives me 14.</think>

<think>But wait, I used 14 already! I need to find another way. Let me try again. I can multiply 40 and 14, which gives me 560.</think>

<think>Now, I need to get 28 by subtracting 532 (560 - 28) from 560. But I don't have 532, I have 40. I can subtract 40 from 560, which gives me 520.</think>

<think>Wait, I made another mistake! I need to get 28, not 520. Let me try again. I can multiply 20 and 14, which gives me 280.</think>

<think>Now, I need to get 48 (28 + 20) by adding 20 to 28. I can do this by dividing 280 by 20, which gives me 14.</think>

<think>But wait, I used 14 already! I need to find another way. Let me try again. I can multiply 40 and 14, which gives me 560.</think>

<think>Now, I need to get 28 by subtracting 532 (560 - 28) from 560. But I don't have 532, I have 40. I can subtract 40 from 560, which gives me 520.</think>

<think>Wait, I made another mistake! I need to get 28, not 520. Let me try again. I can multiply 20 and 14, which gives me 280.</think>

<think>Now, I need to get 48 (28 + 20) by adding 20 to 28. I can do this by dividing 280 by 20, which gives me 14.</think>

<think>But wait, I used 14 already! I need to find another way. Let me try again. I can multiply 40 and 14, which gives me 560.</think>

<think>Now, I need to get 28 by subtracting 532 (560 - 28) from 560. But I don't have 532, I have 40. I can subtract 40 from 560, which gives me 520.</think>

<think>Wait, I made another mistake! I need to get 28, not 520. Let me try again. I can multiply 20 and 14, which gives me 280.</think>

<think>Now, I need to get 48 (28 + 20) by adding 20 to 28. I can do this by dividing 280 by 20, which gives me 14.</think>

<think>But wait, I used 14 already! I need to find another way. Let me try again. I can multiply 40 and 14, which gives me 560.</think>

<think>Now, I need to get 28 by subtracting 532 (560 - 28) from 560. But I don't have 532, I have 40. I can subtract 40 from 560, which gives me 520.</think>

<think>Wait, I made another mistake! I need to get 28, not 520. Let me try again. I can multiply 20 and 14, which gives me 280.</think>

<think>Now, I need to get 48 (28 + 20) by adding 20 to 28. I can do this by dividing 280 by 20, which gives me 14.</think>

<think>But wait, I used 14 already! I need to find another way. Let me try again. I can multiply 40 and 14, which gives me 560.</think>

<think>Now, I need to get 28 by subtracting 532 (560 - 28) from 560. But I don't have 532, I have 40. I can subtract 40 from 560, which gives me 520.</think>

'''

qwen_eval = '''
I need to use the numbers 79, 17, and 60 exactly once to get 36. Let's start by considering the operations we can use. Multiplication and division could potentially give us a large or small number, which might be useful, but addition and subtraction are more straightforward. 
If I divide 60 by 17, I get approximately 3. 529, which is close to 3. 6 but not quite 36. If I divide 60 by 79, I get a very small number, around 0. 766. 
Multiplying 79 by 17 gives me 1343, which is far from 36. 
Adding or subtracting these numbers doesn't seem to get me to 36 either. 
Let's try a different approach. If I subtract 17 from 60, I get 43. Now, if I divide 43 by 79, I get approximately 0. 544, which is still not 36. 
What if I try a different combination? If I subtract 17 from 79, I get 62. Now, if I divide 62 by 17, I get 3. 647, which is close to 3. 6. 
If I then subtract this result from 60, I get 60 - 3. 647 = 56. 353, which is not 36. 
Hmm, this doesn't seem to work. Let's try another combination. If I subtract 17 from 79, I get 62. Now, if I divide 62 by 17, I get 3. 647. If I then subtract this from 60, I get 60 - 3. 647 = 56. 353, which is still not 36. 
I need to rethink this. Let's try a different approach. If I subtract 17 from 60, I get 43. Now, if I divide 43 by 79, I get approximately 0. 544. If I then add this to 17, I get 17 + 0. 544 = 17. 544, which is not 36. 
I need to find a way to get to 36. Let's try another combination. If I subtract 17 from 79, I get 62. Now, if I divide 62 by 17, I get 3. 647. If I then add this to 17, I get 17 + 3. 647 = 20. 647, which is still not 36. 
I need to find a different approach. Let's try a different combination. If I subtract 17 from 60, I get 43. Now, if I divide 43 by 79, I get approximately 0. 544. If I then add this to 17, I get 17 + 0. 544 = 17. 544, which is not 36. 
I need to find a different approach. Let's try a different combination. If I subtract 17 from 79, I get 62. Now, if I divide 62 by 17, I get 3. 647. If I then add this to 17, I get 17 + 3. 647 = 20. 647, which is still not 36. 
I need to find a different approach. Let's try a different combination. If I subtract 17 from 60, I get 43. Now, if I divide 43 by 79, I get approximately 0. 544. If I then add this to 17, I get 17 + 0. 544 = 17. 544, which is not 36. 
I need to find a different approach. Let's try a different combination. If I subtract 17 from 79, I get 62. Now, if I divide 62 by 17, I get 3. 647. If I then add this to 17, I get 17 + 3. 647 = 20. 647, which is still not 36. 
I need to find a different approach. Let
'''

batch = [llama_train, qwen_train, llama_eval, qwen_eval]

for s in batch:
    print('STRING')

    print('lang dist', process_languages(s))
    print()

    # print('lang const reward', language_consistency_reward(s))
    print('lang div reward', language_diversity_reward(s))
    print()

    # print(s)
    # print(semantic_coherence_reward(s))
    # print()

    # print(patch_math(s))
    # print(semantic_coherence_reward(patch_math(s)))

    # print(remove_tags(s))
    # print(semantic_coherence_reward(remove_tags(s)))

    # print(patch_math(remove_tags(s)))
    # print(semantic_coherence_reward(patch_math(remove_tags(s))))