import numpy as np
from plantcv import plantcv as pcv
import cv2

def options():
    """Parse command-line options
    The options function was converted from the class options in Jupyter.
    Rather than hardcoding the inputs, input arguments with the same
    variable names are used to retrieve inputs from the commandline.
    """
    parser = argparse.ArgumentParser(
        description="PlantCV multi-plant workflow")
    parser.add_argument("--image", help="Input image", required=True)
    parser.add_argument("--result", help="Results file", required=True)
    parser.add_argument("--outdir", help="Output directory", required=True)
    parser.add_argument(
        "--writeimg", help="Save output images", action="store_true")
    parser.add_argument("--debug", help="Set debug mode", default=None)
    args = parser.parse_args()
    return args


def main():
    img, path, filename = pcv.readimage(filename=args.image)

    # Convert RGB to HSV and extract the saturation channel
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    print("RGB to HSV")

    # Threshold the saturation image
    s_thresh = pcv.threshold.binary(
        gray_img=s, threshold=85, max_value=255, object_type='light')
    print("Threshold Saturation Image")

    # Median Blur
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    s_cnt = pcv.median_blur(gray_img=s_thresh, ksize=5)
    print("Median Blur")

    # Convert RGB to LAB and extract the Blue channel
    # b = pcv.rgb2gray_lab(gray_img=img, channel='b')
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    print("Extract Blue Channel")

    # Threshold the blue image
    b_thresh = pcv.threshold.binary(
        gray_img=b, threshold=160, max_value=255, object_type='light')
    b_cnt = pcv.threshold.binary(
        gray_img=b, threshold=160, max_value=255, object_type='light')
    print("Threshold Blue Image")

    # Fill small objects
    # b_fill = pcv.fill(b_thresh, 10)

    # Join the thresholded saturation and blue-yellow images
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_cnt)
    print("Join Thresholded saturation and blue-yellow imaes")

    # Apply Mask (for VIS images, mask_color=white)
    # masked = pcv.apply_mask(rgb_img=img, mask=bs, mask_color='white')
    masked = pcv.apply_mask(img=img, mask=bs, mask_color='white')
    print("Apply Mask")
    # Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')
    print("Extract green-magenta and blue-yellow channels")

    # Threshold the green-magenta and blue images
    maskeda_thresh = pcv.threshold.binary(
        gray_img=masked_a, threshold=115, max_value=255, object_type='dark')
    maskeda_thresh1 = pcv.threshold.binary(
        gray_img=masked_a, threshold=135, max_value=255, object_type='light')
    maskedb_thresh = pcv.threshold.binary(
        gray_img=masked_b, threshold=128, max_value=255, object_type='light')
    print("Thresold Green-magenda and blue images")

    # Join the thresholded saturation and blue-yellow images (OR)
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
    print("Join Threshold saturation and blue-yellow images")

    # Fill small objects
    ab_fill = pcv.fill(bin_img=ab, size=200)
    print("Fill small objects")

    # Apply mask (for VIS images, mask_color=white)
    # masked2 = pcv.apply_mask(rgb_img=masked, mask=ab_fill, mask_color='white')
    masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')
    print("Apply 2nd Masked")

    skeleton = pcv.morphology.skeletonize(mask=masked2)
    print("Skeletonize")

    # Identify objects
    id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)
    print("Identify Objects")

    # Define ROI
    roi1, roi_hierarchy = pcv.roi.rectangle(
        img=masked2, x=100, y=100, h=200, w=200)
    print("Define Region of Interest")

    # Decide which objects to keep
    roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,
                                                                   roi_hierarchy=roi_hierarchy,
                                                                   object_contour=id_objects,
                                                                   obj_hierarchy=obj_hierarchy,
                                                                   roi_type='partial')
    print("Decide which objects to keep")

    # Object combine kept objects
    obj, mask = pcv.object_composition(
        img=img, contours=roi_objects, hierarchy=hierarchy3)
    print("Combine kept objects")

    ################################################################

    outfile=False
    if args.writeimg == True:
        outfile = args.outdir + "/" + filename

    # Find shape properties, output shape image (optional)
    shape_imgs = pcv.analyze_object(img=img, obj=obj, mask=mask)

    # Shape properties relative to user boundary line (optional)
    boundary_img1 = pcv.analyze_bound_horizontal(img=img, obj=obj, mask=mask, line_position=1680)

    # Determine color properties: Histograms, Color Slices, output color analyzed histogram (optional)
    color_histogram = pcv.analyze_color(rgb_img=img, mask=kept_mask, hist_plot_type='all')

    # Pseudocolor the grayscale image
    pseudocolored_img = pcv.visualize.pseudocolor(gray_img=s, mask=kept_mask, cmap='jet')

    # Write shape and color data to results file
    pcv.print_results(filename=args.result)


if __name__ == "__main__":
    main()