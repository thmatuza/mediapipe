// Copyright 2020 The MediaPipe Authors.
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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
namespace mediapipe {

namespace {

constexpr int kNumSilhouette = 36;
constexpr int kSilhouette[] = {
  10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109};

constexpr int kNumLipsUpperOuter = 11;
constexpr int kLipsUpperOuter[] = {
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291};
constexpr int kNumLipsLowerOuter = 10;
constexpr int kLipsLowerOuter[] = {
  146, 91, 181, 84, 17, 314, 405, 321, 375, 291};
constexpr int kNumLipsUpperInner = 11;
constexpr int kLipsUpperInner[] = {
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308};
constexpr int kNumLipsLowerInner = 11;
constexpr int kLipsLowerInner[] = {
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308};

constexpr int kNumRightEyeUpper0 = 7;
constexpr int kRightEyeUpper0[] = {
  246, 161, 160, 159, 158, 157, 173};
constexpr int kNumRightEyeLower0 = 9;
constexpr int kRightEyeLower0[] = {
  33, 7, 163, 144, 145, 153, 154, 155, 133};
constexpr int kNumRightEyeUpper1 = 7;
constexpr int kRightEyeUpper1[] = {
  247, 30, 29, 27, 28, 56, 190};
constexpr int kNumRightEyeLower1 = 9;
constexpr int kRightEyeLower1[] = {
  130, 25, 110, 24, 23, 22, 26, 112, 243};
constexpr int kNumRightEyeUpper2 = 7;
constexpr int kRightEyeUpper2[] = {
  113, 225, 224, 223, 222, 221, 189};
constexpr int kNumRightEyeLower2 = 9;
constexpr int kRightEyeLower2[] = {
  226, 31, 228, 229, 230, 231, 232, 233, 244};
constexpr int kNumRightEyeLower3 = 9;
constexpr int kRightEyeLower3[] = {
  143, 111, 117, 118, 119, 120, 121, 128, 245};

constexpr int kNumRightEyebrowUpper = 8;
constexpr int kRightEyebrowUpper[] = {
  156, 70, 63, 105, 66, 107, 55, 193};
constexpr int kNumRightEyebrowLower = 6;
constexpr int kRightEyebrowLower[] = {
  35, 124, 46, 53, 52, 65};

constexpr int kNumRightEyeIris = 5;
constexpr int kRightEyeIris[] = {
  473, 474, 475, 476, 477};

constexpr int kNumLeftEyeUpper0 = 7;
constexpr int kLeftEyeUpper0[] = {
  466, 388, 387, 386, 385, 384, 398};
constexpr int kNumLeftEyeLower0 = 9;
constexpr int kLeftEyeLower0[] = {
  263, 249, 390, 373, 374, 380, 381, 382, 362};
constexpr int kNumLeftEyeUpper1 = 7;
constexpr int kLeftEyeUpper1[] = {
  467, 260, 259, 257, 258, 286, 414};
constexpr int kNumLeftEyeLower1 = 9;
constexpr int kLeftEyeLower1[] = {
  359, 255, 339, 254, 253, 252, 256, 341, 463};
constexpr int kNumLeftEyeUpper2 = 7;
constexpr int kLeftEyeUpper2[] = {
  342, 445, 444, 443, 442, 441, 413};
constexpr int kNumLeftEyeLower2 = 9;
constexpr int kLeftEyeLower2[] = {
  446, 261, 448, 449, 450, 451, 452, 453, 464};
constexpr int kNumLeftEyeLower3 = 9;
constexpr int kLeftEyeLower3[] = {
  372, 340, 346, 347, 348, 349, 350, 357, 465};

constexpr int kNumLeftEyebrowUpper = 8;
constexpr int kLeftEyebrowUpper[] = {
  383, 300, 293, 334, 296, 336, 285, 417};
constexpr int kNumLeftEyebrowLower = 6;
constexpr int kLeftEyebrowLower[] = {
  265, 353, 276, 283, 282, 295};

constexpr int kNumLeftEyeIris = 5;
constexpr int kLeftEyeIris[] = {
  468, 469, 470, 471, 472};

constexpr int kMidwayBetweenEyes = 168;

constexpr int kNoseTip = 1;
constexpr int kNoseBottom = 2;
constexpr int kNoseRightCorner = 98;
constexpr int kNoseLeftCorner = 327;

constexpr int kNumNose = 4;
constexpr int kNose[] = {
  kNoseTip, kNoseLeftCorner, kNoseBottom, kNoseRightCorner};

constexpr int kRightCheek = 205;
constexpr int kLeftCheek = 425;

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kElemsContTag[] = "ELEMS_CONTS";
constexpr char kFaceContTag[] = "FACE_CONTS";

}  // namespace

// A calculator that converts face landmarks to RenderData proto for
// visualization. Ignores landmark_connections specified in
// LandmarksToRenderDataCalculatorOptions, if any, and always uses a fixed set
// of landmark connections specific to face landmark (defined in
// kFaceLandmarkConnections[] above).
//
// Example config:
// node {
//   calculator: "FaceLandmarksToRenderDataCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   output_stream: "RENDER_DATA:render_data"
//   options {
//     [LandmarksToRenderDataCalculatorOptions.ext] {
//       landmark_color { r: 0 g: 255 b: 0 }
//       connection_color { r: 0 g: 255 b: 0 }
//       thickness: 4.0
//     }
//   }
// }
class FaceLandmarksToContoursCalculator
    : public CalculatorBase {
 public:
  static mediapipe::Status GetContract(CalculatorContract* cc);

  mediapipe::Status Open(CalculatorContext* cc) override;
  mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(FaceLandmarksToContoursCalculator);

mediapipe::Status FaceLandmarksToContoursCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kLandmarksTag))
      << "None of the input streams are provided.";

  if (cc->Inputs().HasTag(kLandmarksTag)) {
    cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
  } 
  cc->Outputs().Tag(kElemsContTag).Set<std::vector<NormalizedLandmarkList>>();
  cc->Outputs().Tag(kFaceContTag).Set<std::vector<NormalizedLandmarkList>>();
  return mediapipe::OkStatus();
}

mediapipe::Status FaceLandmarksToContoursCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return mediapipe::OkStatus();
}

mediapipe::Status FaceLandmarksToContoursCalculator::Process(
    CalculatorContext* cc) {
  // Check that landmarks are not empty and skip rendering if so.
  // Don't emit an empty packet for this timestamp.
  if (cc->Inputs().HasTag(kLandmarksTag) &&
      cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
    return mediapipe::OkStatus();
  }

  auto elems_contours =
      absl::make_unique<std::vector<NormalizedLandmarkList>>();

  auto face_contours =
      absl::make_unique<std::vector<NormalizedLandmarkList>>();

  const std::vector<NormalizedLandmarkList>& collection =
          cc->Inputs().Tag(kLandmarksTag).template Get<std::vector<NormalizedLandmarkList>>();

  for (const auto& landmarks : collection) {
    // clock-wise

    NormalizedLandmarkList cnt_left_eye, cnt_right_eye, cnt_nose, cnt_mouth, cnt_face;
    for (int i = 0; i < kNumLeftEyeLower2; ++i) {
      NormalizedLandmark* norm_landmark = cnt_left_eye.add_landmark();
      *norm_landmark = landmarks.landmark(kLeftEyeLower2[kNumLeftEyeLower2 - i - 1]);
    }

    for (int i = 0; i < kNumLeftEyebrowUpper; ++i) {
      NormalizedLandmark* norm_landmark = cnt_left_eye.add_landmark();
      *norm_landmark = landmarks.landmark(kLeftEyebrowUpper[i]);
    }

    elems_contours->emplace_back(cnt_left_eye);

    for (int i = 0; i < kNumRightEyeLower2; ++i) {
      NormalizedLandmark* norm_landmark = cnt_right_eye.add_landmark();
      *norm_landmark = landmarks.landmark(kRightEyeLower2[kNumRightEyeLower2 - i - 1]);
    }

    for (int i = 0; i < kNumRightEyebrowUpper; ++i) {
      NormalizedLandmark* norm_landmark = cnt_right_eye.add_landmark();
      *norm_landmark = landmarks.landmark(kRightEyebrowUpper[i]);
    }

    elems_contours->emplace_back(cnt_right_eye);

    for (int i = 0; i < kNumNose; ++i) {
      NormalizedLandmark* norm_landmark = cnt_nose.add_landmark();
      *norm_landmark = landmarks.landmark(kNose[i]);
    }

    elems_contours->emplace_back(cnt_nose);

    for (int i = 0; i < kNumLipsUpperOuter; ++i) {
      NormalizedLandmark* norm_landmark = cnt_mouth.add_landmark();
      *norm_landmark = landmarks.landmark(kLipsUpperOuter[i]);
    }

    for (int i = 0; i < kNumLipsLowerOuter; ++i) {
      NormalizedLandmark* norm_landmark = cnt_mouth.add_landmark();
      *norm_landmark = landmarks.landmark(kLipsLowerOuter[kNumLipsLowerOuter - i - 1]);
    }

    elems_contours->emplace_back(cnt_mouth);

    for (int i = 0; i < kNumSilhouette; ++i) {
      NormalizedLandmark* norm_landmark = cnt_face.add_landmark();
      *norm_landmark = landmarks.landmark(kSilhouette[i]);
    }

    face_contours->emplace_back(cnt_face);
  }

  cc->Outputs()
      .Tag(kElemsContTag)
      .Add(elems_contours.release(), cc->InputTimestamp());

  cc->Outputs()
      .Tag(kFaceContTag)
      .Add(face_contours.release(), cc->InputTimestamp());
  return mediapipe::OkStatus();
}

} 