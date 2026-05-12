#calculate pixel differential
import numpy, math
x1, y1 = (1416, 6) #pixels   <-- move delta Y one
x2, y2 = (1213, 2040) #pixels <-- move delta Y one
x1, y1 = (427, 4)  #<-- move delta Y 2    
x2, y2 = (237, 2024) #<-- move delta Y 2

x1, y1 = (392, 785)  #<-- move delta x 1
x2, y2 = (2449, 972) #<-- move delta x 1
microns_per_pixel = 3.45 #pixel pitch in both directions for a2A2464-77umBAS
mm_per_micron = 1/1000  #
magnification = 1/20 #1/10 # Nee
stage_traveled_distance = 0.33854#delta x 0.641 #0.632 delta y2 #0.636 delta y1 
x1, y1 = (2200, 1045)
x2, y2 = (10, 846)

'''
TO DO:
    - Use CV2 to identify where the slit is in the image; by feed it the very most LEFT columns of the HSI cube and have it match the pattern in RGB
            --> this should solve for translation right away good to know where slit is relative to each FOV
            
    - Do a reconstruction of each FOV using below distances and settings:
        stage movement = delta x ; 
        
        
        A. 3.45 pixels per micron x 2448 pixels on RGB camera /10.78 adjusted FOV microns per micron =  783.45 --> 783.3 microns needed to transverse X FOV in theory
        
        B. From other data; (1/speed) * (distance) ≈ (# of lines) * (integration time / line ) ; acceleration is 50 ms ~ 3 lines; linear acceleration so ~1.5 line error
                --> empirically found: (1/0.0623 mm/s) * (0.845 mm) ≈ (834 lines)(16.33 ms)
                                        
                                        #13.56 secs ≈ 13619 ms ; 13.56 secs ≈ 13.62 secs  ✅ this is exact expectation 
                                        with acceleration error on stage; 13619 + (16.33)*1.5 /1000 = 13.64 ms
                                        ***#via python clocking include some code runtime: 🕒[checking x_movement] elapsed 13.738908s (stopped at line 4059)
                                            13.64 ms ≈ 13.62 secs  ✅ <-- slight polling error on camera side so maybe not perfect 1-2 lines of error itself
                                            
                                                --> 1 line is therefore 834 lines / time --> (1/0.0623) * (x) = (1)*(16.33) <--> x = 0.001017 microns
                                                    --> 1 line = 1.017 microns roughly speak; it technically varies per calculation bc of error
                                                    previous calculation it seemed to be ~1.03 ; this sizing though matches the slit width / magnification
                                                    or the short direction of the slit almost perfectly 
                                                    
        C. So if want to match cameras FOV from (A) & (B) we have --> then we need to scan 783 microns / (1 line / 1.02 microns) = 767.64 --> 768 lines 
                ➪ if I scan at 0.783 mm I should get a good POV with 768 lines with little to no FOV discrepancy
  
                - Results in 2x2 scan: 
                       +x     -x
                       772    764
                       772    766 lines  which matches the ideal target well +- 4 lines bound
                       
                - Reality is theres a rotation or shear so we would have to travel cos(theta) of shear or match the 1/cot shear in delta x dir maybe 5 degrees
                            ➪ i.e. we need to travel 0.783 * cos(theta) where theta is rotation angle
                           
                           -When moving 1 FOV in the RGB camera and manually measuring pixels:
                           
                            x1, y1 = (392, 785)  #<-- move delta x 1
                            x2, y2 = (2449, 972) #<-- move delta x 1
                            microns_per_pixel = 3.45 #pixel pitch in both directions for a2A2464-77umBAS
                            mm_per_micron = 1/1000  #
                            magnification = 1/20 #1/10 # Nee
                            stage_traveled_distance = 0.33854#delta x 0.641 #0.632 delta y2 #0.636 delta y1 
                            
                            Trial results (angle error between stage axes and camera axes):

                            Trial 1 (Y motion): 5.7°
                            Trial 2 (Y motion): 5.4°
                            Trial 3 (X motion): 5.2°

                            Average Y-axis misalignment ≈ 5.55°
                            Average X-axis misalignment ≈ 5.2°

                            Best single estimate of stage rotation (axis skew):
                            theta ≈ 5.5°

                            Small-angle equivalent coupling coefficient:
                            k ≈ tan(5.5°) ≈ 0.096
                    so we should effectively move measured_x = 0.784 / cos(5.5°) ≈ .788 mm to get same FOV
                        --> thats 775 lines
        '''


#Thought to be FOV on 10x mag for both cameras according to ASI stage:
    #a2A1920-155um HSI camera  :  (x = variable, y = 0.845 mm); each dim correlates with slit length ; pixel pitch = 3.45 μm 
    #a2A2464-77umBAS RGB camera:  (x = 0.712 y = 0.642, ); static field of view; pixel pitch = 5.86 μm 
    
#looks like the camera distance traveled is 10% larger (1.1x bigger) than the stage distance traveled. 
#This is consistent across both cameras; 
    # i.e. RGB a2A2464-77umBAS camera y component measured on the stage top via pixel tracking = 0.712 mm 
#which aligns extremely well to the expected sensor dim of 7.07 according to the website 
    #whereas the ASI stagetop measurement on on 10x magnification = is 0.642 mm ; 
    # 0.712 / 0.642 ~1.11
    # 0.707 / 0.642 ~ 1.10 ... either comparison empirical sensor :: empirical stage, or theoretical sensor :: empirical stage
        #The error ratio is about the same
        
    # on the a2A1920-155um HSI camera ; the delta y (non-scanning spatial direction relative to stage basis) 
# effectively is measured as 1 line = [# of spatial,  # spectral pixles]  --> 
# the number of spatial pixels per line corresponds to: slit length/magnification --> #slit was mentioned to be 1.02 mm online so probably some culimating lense in back
        # ~1600 x 5.86 microns / 10x mag * 1 mm / 1000 micron = 0.932 microns
        # when empirically testing hot-cold for write delta y = 0.845 microns
        # mismatch is 0.932/0.845 = 1.10; 
        # ∴ it looks like theres an issue with the stage calibration distance as a func what seems to be speed as we
        # the acceleration the stage period of stage is set to 50 ms for HSI camera data and not set for RBG camera; eliminated possibiltiy
        # the distance traveled to test error ratio was different; would be a CONSTANT error not a scalar term
            # --> tested this by putting light weight ruler not heavy enough in itself to cause error on the stage
            # parallel roughly to the x direction
            # put the light cone approixmately center of 3 cm mark; traveled to 12 cm mark. This is 9 cm according to ruler.
            # According to the stage it's 89.9 mm = 9 cm; good match could just me be wrong by not eyeballing center of right correctly
                #same thing with second ruler second different ruler ~ 89.9 mm differential with varing speed/accel alon the way
        
        # therefore the problem was must be the speed of the stage itself or the optics will test on 20x mag on RGB to see if same issue
        
        #(2200, 1045) #(10, 846); delta x = 0.33854 stage; calculated expected is 0.379 mm across hypotenuse 
                    # --> 0.379 mm / 0.33854 = 1.12
#solve for y component;
pixels_traveled: int = int(math.sqrt(((x1 - x2)**2 + (y1 - y2)**2)) + 0.5) #approx round to nearest pixel
print(pixels_traveled)

scalar_factor: float = microns_per_pixel *  magnification * mm_per_micron
microns_traveled: float = microns_per_pixel * pixels_traveled 
hypotenuse_mm_traveled: float = mm_per_micron * microns_traveled * magnification  #
print(f'total mm traveled along the hypotenuse is {hypotenuse_mm_traveled}')

other_component = lambda a, b, c, d=0: c**2 if (a > c) else (a**2 - b**2)  
'''
def calc_side_right_triangle(*args):
    a, b, c = args
    d = 
    assert (a + b > c) and all((type(item) is type(int(item))) for item in args if item != 0)  # fix: list comprehension wasn't being evaluated as a bool; wrapped in all()
    # fix: swap logic — find the largest side (hypotenuse) and ensure it's in position c
    a, b, c = sorted(args)  # sort so c is always the largest
    assert (c > a) and (c > b)
    return math.sqrt(c**2 - a**2)  
calc_side_right_triangle( a -)
'''

def calc_side_right_triangle(*args):
    '''assume c is 0th element... handles any number of vars such that element1^2 + element2^2 + element3^2 + ... = c^2'''
    # args[0] is always treated as the hypotenuse (c)
    c, *rest = args

    if c is None or c == 0:
        # c unknown: sqrt of sum of all other sides squared
        assert all(v is not None and v != 0 for v in rest), "Only one unknown allowed"
        return math.sqrt(sum(v**2 for v in rest))
    else:
        # one of the legs is unknown: sqrt(c² - sum of known legs²)
        unknowns = [v for v in rest if v is None or v == 0]
        knowns   = [v for v in rest if v is not None and v != 0]
        assert len(unknowns) == 1, "Exactly one unknown required"
        return math.sqrt(c**2 - sum(v**2 for v in knowns))
    
print(f'Calculated other component is: {calc_side_right_triangle(hypotenuse_mm_traveled, stage_traveled_distance, None)} mm')
print(f'Expected other component is {abs((x1 - x2))* scalar_factor}')

