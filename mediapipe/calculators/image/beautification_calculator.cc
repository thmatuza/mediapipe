// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_gapi_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace {
constexpr char kElemsContTag[] = "ELEMS_CONTS";
constexpr char kFaceContTag[] = "FACE_CONTS";
constexpr char kImageFrameTag[] = "IMAGE";
}  // namespace

namespace config
{
const     cv::Scalar kClrWhite (255, 255, 255);
const     cv::Scalar kClrGreen (  0, 255,   0);
const     cv::Scalar kClrYellow(  0, 255, 255);

const     cv::Size   kGKernelSize(5, 5);
constexpr double     kGSigma       = 0.0;
constexpr int        kBSize        = 9;
constexpr double     kBSigmaCol    = 30.0;
constexpr double     kBSigmaSp     = 30.0;
constexpr int        kUnshSigma    = 3;
constexpr float      kUnshStrength = 0.7f;
constexpr int        kAngDelta     = 1;
constexpr bool       kClosedLine   = true;

} // namespace config

namespace {
using Contour   = std::vector<cv::Point>;
}

namespace custom
{
G_TYPED_KERNEL(GBilatFilter, <cv::GMat(cv::GMat,int,double,double)>,
               "custom.faceb12n.bilateralFilter")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, int,double,double)
    {
        return in;
    }
};

G_TYPED_KERNEL(GLaplacian, <cv::GMat(cv::GMat,int)>,
               "custom.faceb12n.Laplacian")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, int)
    {
        return in;
    }
};

G_TYPED_KERNEL(GFillPolyGContours, <cv::GMat(cv::GMat,cv::GArray<Contour>)>,
               "custom.faceb12n.fillPolyGContours")
{
    static cv::GMatDesc outMeta(cv::GMatDesc in, cv::GArrayDesc)
    {
        return in.withType(CV_8U, 1);
    }
};

// OCV_Kernels
// This kernel applies Bilateral filter to an input src with default
//  "cv::bilateralFilter" border argument
GAPI_OCV_KERNEL(GCPUBilateralFilter, custom::GBilatFilter)
{
    static void run(const cv::Mat &src,
                    const int      diameter,
                    const double   sigmaColor,
                    const double   sigmaSpace,
                          cv::Mat &out)
    {
        cv::bilateralFilter(src, out, diameter, sigmaColor, sigmaSpace);
    }
};

// This kernel applies Laplace operator to an input src with default
//  "cv::Laplacian" arguments
GAPI_OCV_KERNEL(GCPULaplacian, custom::GLaplacian)
{
    static void run(const cv::Mat &src,
                    const int      ddepth,
                          cv::Mat &out)
    {
        cv::Laplacian(src, out, ddepth);
    }
};

// This kernel draws given white filled contours "cnts" on a clear Mat "out"
//  (defined by a Scalar(0)) with standard "cv::fillPoly" arguments.
//  It should be used to create a mask.
// The input Mat seems unused inside the function "run", but it is used deeper
//  in the kernel to define an output size.
GAPI_OCV_KERNEL(GCPUFillPolyGContours, custom::GFillPolyGContours)
{
    static void run(const cv::Mat              &,
                    const std::vector<Contour> &cnts,
                          cv::Mat              &out)
    {
        out = cv::Scalar(0);
        cv::fillPoly(out, cnts, config::kClrWhite);
    }
};

// GAPI subgraph functions
inline cv::GMat unsharpMask(const cv::GMat &src,
                            const int       sigma,
                            const float     strength);
inline cv::GMat mask3C(const cv::GMat &src,
                       const cv::GMat &mask);
} // namespace custom

//! [unsh]
inline cv::GMat custom::unsharpMask(const cv::GMat &src,
                                    const int       sigma,
                                    const float     strength)
{
    cv::GMat blurred   = cv::gapi::medianBlur(src, sigma);
    cv::GMat laplacian = custom::GLaplacian::on(blurred, CV_8U);
    return (src - (laplacian * strength));
}
//! [unsh]

inline cv::GMat custom::mask3C(const cv::GMat &src,
                               const cv::GMat &mask)
{
    std::tuple<cv::GMat,cv::GMat,cv::GMat> tplIn = cv::gapi::split3(src);
    cv::GMat masked0 = cv::gapi::mask(std::get<0>(tplIn), mask);
    cv::GMat masked1 = cv::gapi::mask(std::get<1>(tplIn), mask);
    cv::GMat masked2 = cv::gapi::mask(std::get<2>(tplIn), mask);
    return cv::gapi::merge3(masked0, masked1, masked2);
}

namespace mediapipe {

// A calculator to recolor a masked area of an image to a specified color.
//
// A mask image is used to specify where to overlay a user defined color.
// The luminance of the input image is used to adjust the blending weight,
// to help preserve image textures.
//
// Inputs:
//   One of the following IMAGE tags:
//   IMAGE: An ImageFrame input image, RGB or RGBA.
//   IMAGE_GPU: A GpuBuffer input image, RGBA.
//   One of the following MASK tags:
//   MASK: An ImageFrame input mask, Gray, RGB or RGBA.
//   MASK_GPU: A GpuBuffer input mask, RGBA.
// Output:
//   One of the following IMAGE tags:
//   IMAGE: An ImageFrame output image.
//   IMAGE_GPU: A GpuBuffer output image.
//
// Options:
//   color_rgb (required): A map of RGB values [0-255].
//   mask_channel (optional): Which channel of mask image is used [RED or ALPHA]
//
// Usage example:
//  node {
//    calculator: "BeautificationCalculator"
//    input_stream: "IMAGE_GPU:input_image"
//    input_stream: "MASK_GPU:input_mask"
//    output_stream: "IMAGE_GPU:output_image"
//    node_options: {
//      [mediapipe.BeautificationCalculatorOptions] {
//        color { r: 0 g: 0 b: 255 }
//        mask_channel: RED
//      }
//    }
//  }
//
// Note: Cannot mix-match CPU & GPU inputs/outputs.
//       CPU-in & CPU-out <or> GPU-in & GPU-out
class BeautificationCalculator : public CalculatorBase {
 public:
  BeautificationCalculator() : pipeline([=]()
  {
//! [net_usg_fd]
      cv::GMat  gimgIn;                                                                           // input

//! [net_usg_ld]
      cv::GArray<Contour> garElsConts;                                                            // face elements
      cv::GArray<Contour> garFaceConts;                                                           // whole faces

//! [msk_ppline]
      cv::GMat mskSharp        = custom::GFillPolyGContours::on(gimgIn, garElsConts);             // |
      cv::GMat mskSharpG       = cv::gapi::gaussianBlur(mskSharp, config::kGKernelSize,           // |
                                                        config::kGSigma);                         // |
      cv::GMat mskBlur         = custom::GFillPolyGContours::on(gimgIn, garFaceConts);            // |
      cv::GMat mskBlurG        = cv::gapi::gaussianBlur(mskBlur, config::kGKernelSize,            // |
                                                        config::kGSigma);                         // |draw masks
      // The first argument in mask() is Blur as we want to subtract from                         // |
      // BlurG the next step:                                                                     // |
      cv::GMat mskBlurFinal    = mskBlurG - cv::gapi::mask(mskBlurG, mskSharpG);                  // |
      cv::GMat mskFacesGaussed = mskBlurFinal + mskSharpG;                                        // |
      cv::GMat mskFacesWhite   = cv::gapi::threshold(mskFacesGaussed, 0, 255, cv::THRESH_BINARY); // |
      cv::GMat mskNoFaces      = cv::gapi::bitwise_not(mskFacesWhite);                            // |
//! [msk_ppline]

      cv::GMat gimgBilat       = custom::GBilatFilter::on(gimgIn, config::kBSize,
                                                          config::kBSigmaCol, config::kBSigmaSp);
      cv::GMat gimgSharp       = custom::unsharpMask(gimgIn, config::kUnshSigma,
                                                      config::kUnshStrength);
      // Applying the masks
      // Custom function mask3C() should be used instead of just gapi::mask()
      //  as mask() provides CV_8UC1 source only (and we have CV_8U3C)
      cv::GMat gimgBilatMasked = custom::mask3C(gimgBilat, mskBlurFinal);
      cv::GMat gimgSharpMasked = custom::mask3C(gimgSharp, mskSharpG);
      cv::GMat gimgInMasked    = custom::mask3C(gimgIn,    mskNoFaces);
      cv::GMat gimgBeautif = gimgBilatMasked + gimgSharpMasked + gimgInMasked;
      return cv::GComputation(cv::GIn(gimgIn, garFaceConts, garElsConts), cv::GOut(gimgBeautif, cv::gapi::copy(gimgIn)));
  }){}
  ~BeautificationCalculator() override = default;

  static mediapipe::Status GetContract(CalculatorContract* cc);

  mediapipe::Status Open(CalculatorContext* cc) override;
  mediapipe::Status Process(CalculatorContext* cc) override;
  mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  mediapipe::Status LoadOptions(CalculatorContext* cc);
  mediapipe::Status RenderCpu(CalculatorContext* cc);
  void ConvertContours(const std::vector<NormalizedLandmarkList>& contours, std::vector<Contour>& vctConts, int image_width, int image_height);

  bool initialized_ = false;
  std::vector<float> color_;

  cv::GComputation pipeline;
  cv::GCompiled compiled;
  cv::gapi::GKernelPackage customKernels;
  cv::gapi::GKernelPackage kernels;
};
REGISTER_CALCULATOR(BeautificationCalculator);

// static
mediapipe::Status BeautificationCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kElemsContTag)) {
    cc->Inputs().Tag(kElemsContTag).Set<std::vector<NormalizedLandmarkList>>();
  }

  if (cc->Inputs().HasTag(kFaceContTag)) {
    cc->Inputs().Tag(kFaceContTag).Set<std::vector<NormalizedLandmarkList>>();
  }

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }

  if (cc->Outputs().HasTag(kImageFrameTag)) {
    cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }

  // Confirm only one of the input streams is present.
  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag));
  // Confirm only one of the output streams is present.
  RET_CHECK(cc->Outputs().HasTag(kImageFrameTag));

  return mediapipe::OkStatus();
}

mediapipe::Status BeautificationCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  // Declaring custom and fluid kernels have been used:
  //! [kern_pass_1]
  customKernels = cv::gapi::kernels<custom::GCPUBilateralFilter,
                                           custom::GCPULaplacian,
                                           custom::GCPUFillPolyGContours>();
  kernels      = cv::gapi::combine(cv::gapi::core::fluid::kernels(),
                                           customKernels);
  //! [kern_pass_1]

  return mediapipe::OkStatus();
}

mediapipe::Status BeautificationCalculator::Process(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(RenderCpu(cc));
  return mediapipe::OkStatus();
}

mediapipe::Status BeautificationCalculator::Close(CalculatorContext* cc) {
  return mediapipe::OkStatus();
}

void BeautificationCalculator::ConvertContours(
  const std::vector<NormalizedLandmarkList>& contours,
  std::vector<Contour>& vctConts, int image_width, int image_height) {
  Contour cnt;
  for (const auto& landmarks : contours) {
    cnt.clear();
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
      const NormalizedLandmark& landmark = landmarks.landmark(i);
      cv::Point pt(landmark.x() * image_width, landmark.y() * image_height);
      cnt.push_back(pt);
    }
    vctConts.push_back(cnt);
  }
}

mediapipe::Status BeautificationCalculator::RenderCpu(CalculatorContext* cc) {
  // Get inputs and setup output.
  const auto& input_img = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
  std::vector<Contour> vctFaceConts, vctElsConts;
  if (cc->Inputs().HasTag(kElemsContTag) &&
          !cc->Inputs().Tag(kElemsContTag).IsEmpty()) {
    const auto& elems_contours = cc->Inputs().Tag(kElemsContTag).Get<std::vector<NormalizedLandmarkList>>();
    ConvertContours(elems_contours, vctElsConts, input_img.Width(), input_img.Height());
  }
  if (cc->Inputs().HasTag(kFaceContTag) &&
          !cc->Inputs().Tag(kFaceContTag).IsEmpty()) {
    const auto& face_contours = cc->Inputs().Tag(kFaceContTag).Get<std::vector<NormalizedLandmarkList>>();
    ConvertContours(face_contours, vctFaceConts, input_img.Width(), input_img.Height());
  }
  cv::Mat input_mat = formats::MatView(&input_img);
  cv::Mat img;
  cv::cvtColor(input_mat, img, cv::COLOR_BGRA2BGR);

  cv::Mat imgShow;
  cv::Mat imgBeautif;
  if (!compiled)
  {
    compiled = pipeline.compile(cv::descr_of(cv::gin(img, vctFaceConts, vctElsConts)), cv::compile_args(kernels));
  }
#if 1
  compiled(cv::gin(img, vctFaceConts, vctElsConts), cv::gout(imgBeautif, imgShow));

  auto output_img = absl::make_unique<ImageFrame>(
      input_img.Format(), input_mat.cols, input_mat.rows);
  cv::Mat output_mat = mediapipe::formats::MatView(output_img.get());

  /*cv::polylines(imgShow, vctFaceConts, config::kClosedLine,
                config::kClrYellow);
  cv::polylines(imgShow, vctElsConts, config::kClosedLine,
                config::kClrYellow);*/
  cv::cvtColor(imgShow, output_mat, cv::COLOR_BGR2BGRA);
  //cv::cvtColor(imgBeautif, output_mat, cv::COLOR_BGR2BGRA);
#else
  auto output_img = absl::make_unique<ImageFrame>(
      input_img.Format(), input_mat.cols, input_mat.rows);
  cv::Mat output_mat = mediapipe::formats::MatView(output_img.get());
  cv::cvtColor(img, output_mat, cv::COLOR_BGR2BGRA);
#endif
  cc->Outputs()
      .Tag(kImageFrameTag)
      .Add(output_img.release(), cc->InputTimestamp());
  return mediapipe::OkStatus();
}

}  // namespace mediapipe