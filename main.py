# **************************************************************************
# *
# * Authors: Alberto Garcia Mena   (alberto.garcia@cnb.csic.es)
# *
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


###
#Requirements
#pip install -qU langchain-huggingface

dictProtocols = {} #text/vector, text/phrases
dictProtocolSummary = {}
questionForProtocols= 'Describe everything this Scipion protocol does. First, provide a summary (200 words) with the main keywords. Then, explain what does all the parameter (defineParameters) (200 words). Finally, describe the inputs and outputs (200 words). Omit any tittle in the three steps: \n'
userQuestion1= 'to align movies'
userQuestion2 = 'to manage holes in Smartscope'
userQuestion3 = 'to normalize particles'

splittersSummary1 = 'defineParameters'
splittersSummary2 = 'Inputs and Outputs'


######COLLECT SUMMARIES
dictProtocolSummary['flexAlign'] = '''The XmippProtFlexAlign protocol is a Scipion-based tool designed for movie alignment in cryo-electron microscopy (cryo-EM) using cross-correlation. It supports both global and local alignment, with optional GPU acceleration. Key features include handling EER movies, automatic control point determination, and patch-based local alignment. The protocol also computes Power Spectral Density (PSD) for quality assessment and allows gain reference transformation. It is optimized for parallel processing and integrates with Xmipp's GPU and CPU implementations.

The defineParameters function configures the protocol's parameters. Key parameters include:

    nFrames: Number of frames for EER movies, grouping subframes.

    doLocalAlignment: Enables local alignment, similar to MotionCor2.

    autoControlPoints: Automatically determines control points for BSpline interpolation.

    controlPointX/Y/T: Manual control points for X, Y, and time dimensions.

    autoPatches: Automatically determines patches for local alignment.

    patchesX/Y: Manual patches for local alignment.

    minLocalRes: Minimum patch size in Angstroms.

    groupNFrames: Groups frames for alignment by summing them.

    maxResForCorrelation: Maximum resolution preserved during correlation.

    doComputePSD: Computes PSD before and after alignment.

    maxShift: Maximum allowed frame shift in Angstroms.

    gainRot/gainFlip: Rotates and flips the gain reference.

    GPU_LIST: Specifies GPU devices for acceleration.

    USE_GPU: Toggles between GPU and CPU implementations.

Inputs and Outputs:

    Inputs:

        inputMovies: A set of movies to align, which can include EER or other formats.

        Dark and Gain references: Optional dark and gain references for correction.

        Frame ranges: Specifies frames for alignment and summation.

    Outputs:

        Aligned movies: Movies corrected for global and local motion.

        Averaged micrographs: Summed frames from aligned movies.

        PSD images: Power Spectral Density images for quality assessment.

        Shift plots: Visualizations of frame shifts during alignment.

        Shift metadata: Files containing shift values for each frame.

The protocol generates intermediate files, such as control points, patches, and transformed gain references, which are used during processing. Outputs are stored in Scipion's project structure for further analysis.'''
dictProtocolSummary['motionCor'] = '''The ProtMotionCorr protocol in Scipion is designed for movie alignment in cryo-EM using the MotionCor2 software. It corrects anisotropic drift and applies dose weighting to movies, enabling high-quality micrograph generation. Key features include frame range selection, binning, cropping, and optional splitting of even/odd frames for validation. The protocol supports GPU acceleration, computes Power Spectral Density (PSD) for quality assessment, and generates micrograph thumbnails. It also provides detailed shift plots for alignment visualization and integrates with EMAN2 for PSD and thumbnail computation.

The defineParameters function configures the protocol's parameters. Key parameters include:

    alignFrame0/N: Specifies the range of frames to align and sum.

    binFactor: Binning factor (1x or 2x) applied before processing.

    cropOffsetX/Y: Cropping offsets in pixels.

    cropDimX/Y: Cropping dimensions in pixels.

    splitEvenOdd: Splits and sums odd/even frames for validation.

    doSaveMovie: Saves the aligned movie.

    doComputePSD: Computes PSD for aligned micrographs.

    doComputeMicThumbnail: Generates micrograph thumbnails.

    extraProtocolParams: Additional protocol-specific parameters.

    doseFiltering: Applies dose filtering to correct for beam-induced motion.

    patchX/Y: Enables patch-based alignment for large movies.

    gainRot/gainFlip: Rotates and flips the gain reference.

Inputs and Outputs:

    Inputs:

        inputMovies: A set of movies to align, which can include formats like MRC, TIFF, or EER.

        Dark and Gain references: Optional dark and gain references for correction.

        Frame ranges: Specifies frames for alignment and summation.

    Outputs:

        Aligned movies: Movies corrected for global and local motion.

        Averaged micrographs: Summed frames from aligned movies, optionally dose-weighted.

        PSD images: Power Spectral Density images for quality assessment.

        Shift plots: Visualizations of frame shifts during alignment.

        Micrograph thumbnails: Thumbnails for quick visualization.

        Even/Odd micrographs: Separate sums for odd and even frames if enabled.

The protocol generates intermediate files, such as alignment logs and shift metadata, which are used during processing. Outputs are stored in Scipion's project structure for further analysis.'''
dictProtocolSummary['centerParticle'] ='''The XmippProtCenterParticles protocol in Scipion is designed to realign and center particles within 2D classes. It takes a set of 2D classes and their associated micrographs as input, computes the center of mass for each particle, and applies transformations to align them. The protocol outputs a new set of centered particles and updated 2D classes. Key features include particle realignment, transformation matrix computation, and generation of a summary file detailing the centering process. It is particularly useful for improving the alignment of particles before further processing, such as 3D reconstruction.

The defineParameters function configures the protocol's parameters. Key parameters include:

    inputClasses: A set of 2D classes to be realigned and centered.

    inputMics: The set of micrographs associated with the input classes.

    parallelSection: Configures parallel processing options (threads and MPI).

The protocol does not expose additional advanced parameters, as it primarily relies on the input classes and micrographs to perform the centering operation. The transformation matrices and particle coordinates are computed internally based on the alignment data stored in the input classes.

Inputs and Outputs:

    Inputs:

        inputClasses: A set of 2D classes containing particles to be realigned.

        inputMics: The set of micrographs associated with the input classes, used to map particle coordinates.

    Outputs:

        outputClasses: A new set of 2D classes with centered particles.

        outputParticles: A set of particles with updated coordinates and transformations.

        summary.txt: A text file summarizing the centering process, including the percentage of particles centered and the average displacement.

The protocol generates intermediate files, such as metadata files for centered particles and transformation matrices, which are used during processing. The final outputs are stored in Scipion's project structure for further analysis.'''
dictProtocolSummary['feedBackFilter'] ='''The smartscopeFeedbackFilter protocol in Scipion is designed to provide feedback to Smartscope during data acquisition in cryo-EM. It analyzes micrographs filtered by alignment, CTF estimation, and other criteria to determine the best holes for further acquisition. The protocol calculates statistics on hole quality, identifies holes that pass or fail filters, and sends intensity range recommendations back to Smartscope. Key features include real-time feedback, hole classification, and intensity range optimization. It operates in streaming mode, continuously updating its analysis as new micrographs are acquired. The protocol outputs sets of holes that pass or fail the filters and generates summary statistics for visualization.

The defineParameters function configures the protocol's parameters. Key parameters include:

    inputProtocol: The Smartscope connection protocol providing the input data.

    micsPassFilter: A set of micrographs filtered by alignment or CTF estimation.

    triggerMicrograph: The number of micrographs required to launch the protocol.

    emptyBinsPercent: The allowed percentage of empty bins in the intensity histogram.

    multishotThreshold: The percentage of quality-filtered shots required for a hole to pass.

    refreshTime: The time interval (in seconds) for refreshing data and updating feedback.

    parallelSection: Configures parallel processing options (threads and MPI).

These parameters allow the protocol to adapt to different acquisition scenarios and filter criteria, ensuring optimal feedback to Smartscope.

Inputs and Outputs:

    Inputs:

        inputProtocol: The Smartscope connection protocol providing grids, holes, and movies.

        micsPassFilter: A set of micrographs filtered by alignment, CTF estimation, or other criteria.

    Outputs:

        SetOfHolesPassFilter: A set of holes that pass the quality filters.

        SetOfHolesRejected: A set of holes that fail the quality filters.

        summary.txt: A text file summarizing the analysis, including intensity ranges and statistics.

        Viewer files: Histogram data files for visualizing hole intensity distributions.

The protocol generates intermediate files, such as intensity histograms and hole classifications, which are used during processing. The final outputs are stored in Scipion's project structure for further analysis and visualization. The protocol also communicates directly with Smartscope to update acquisition parameters in real-time.'''
dictProtocolSummary['exportCoord'] ='''Summary (200 words):
The ProtExportCoordinates3D protocol in Scipion is designed to export 3D subtomogram coordinates for use outside the Scipion framework. It supports multiple output formats, including plain text (txt), RELION (star), EMAN2 (eman), Dynamo (dynamo), and SPHIRE (cbox). The protocol takes a SetOfCoordinates3D as input, which contains 3D coordinates mapped to tomograms, and exports them in the specified format. The exported files can be used for further processing in external software packages. The protocol is flexible, allowing users to choose the desired output format based on the tools they intend to use. It handles coordinate transformations and alignment information, ensuring compatibility with external workflows. The protocol is part of the Scipion ecosystem and integrates with plugins like RELION, EMAN2, Dynamo, and SPHIRE, making it a versatile tool for cryo-ET data processing. It is particularly useful for users who need to transfer coordinate data between different software platforms for tasks such as subtomogram averaging or particle picking.

Parameters (200 words):
The defineParams function defines two main parameters. The first is inputCoordinates, a PointerParam that points to a SetOfCoordinates3D object, which contains the 3D coordinates to be exported. This parameter is marked as important, as it is the primary input for the protocol. The second parameter is outputFormat, an EnumParam that allows users to select the export format. The choices for this parameter are dynamically generated based on the available plugins (e.g., RELION, EMAN2, Dynamo, SPHIRE). By default, the protocol exports to plain text (txt), but users can choose other formats if the corresponding plugins are installed. The outputFormat parameter ensures that the exported coordinates are compatible with the desired external software. Additional parameters may be inferred from the context, such as the sampling rate of the input coordinates, which is used to scale the exported data appropriately. The protocol also handles coordinate transformations, ensuring that alignment information is preserved in the exported files.

Inputs and Outputs (200 words):
The input to the protocol is a SetOfCoordinates3D object, which contains 3D coordinates mapped to specific tomograms. These coordinates typically represent the positions of particles or regions of interest within the tomograms. The protocol processes these coordinates and exports them in the selected format. The output format can be plain text (txt), RELION (star), EMAN2 (eman), Dynamo (dynamo), or SPHIRE (cbox), depending on the user's choice. For plain text, the output is a simple text file with coordinates listed as x y z values. For RELION, the output is a .star file containing coordinate metadata. For EMAN2, the output is a JSON file with coordinate information. For Dynamo, the output is a .tbl file with additional alignment data. For SPHIRE, the output is a .cbox file. The exported files are saved in a dedicated Export directory within the protocol's working directory. The protocol ensures that the exported coordinates are compatible with the target software, making it easier to continue processing outside Scipion. The output files can be directly used in external workflows for tasks like subtomogram averaging or particle refinement.'''
dictProtocolSummary['preprocessParticle'] ='''Summary (200 words):
The Scipion protocol, XmippProtPreprocessParticles and XmippProtPreprocessVolumes, is designed for preprocessing particles and volumes in cryo-EM data analysis. Key functionalities include dust removal, normalization, contrast inversion, thresholding, phase flipping, and randomization of phases. For particles, it supports centering, phase flipping, and normalization with options for background adjustment. For volumes, it includes symmetrization, Laplacian denoising, segmentation, and gray value adjustment. The protocol allows GPU acceleration and provides advanced options for thresholding, normalization, and segmentation. It is part of the Xmipp software suite and is used to prepare data for further processing, such as 3D reconstruction. The protocol is highly customizable, with parameters for GPU usage, thresholding methods, normalization types, and segmentation techniques. It outputs processed particles or volumes, ready for downstream analysis.

Parameters (200 words):
The defineParameters function defines several key parameters for preprocessing. For particles, parameters include doRemoveDust (removes outliers), doNormalize (normalizes background), doInvert (inverts contrast), doThreshold (applies thresholding), and doPhaseFlip (flips phases). Thresholding options include thresholdType (abs_below, below, above), fillType (value, binarize, avg), and fillValue (value for substitution). For volumes, parameters include doChangeHand (mirrors volume), doRotateIco (rotates icosahedral volumes), doSymmetrize (applies symmetry), doLaplacian (denoises), and doSegment (segments volumes). Segmentation options include segmentationType (voxel mass, aminoacid mass, dalton mass, automatic) and segmentationMass (mass value). Normalization parameters include backRadius (background radius) and normType (OldXmipp, NewXmipp, Ramp). GPU usage is controlled by USE_GPU and GPU_LIST. Additional parameters include doRandomize (randomizes phases) and doAdjust (adjusts gray values). These parameters allow fine-tuning of preprocessing steps to suit specific data requirements.

Inputs and Outputs (200 words):
The protocol takes as input a set of particles (SetOfParticles) or volumes (Volume or SetOfVolumes). For particles, the input includes metadata and image stacks, while for volumes, it includes 3D maps. Additional inputs may include masks (VolumeMask) for symmetrization or denoising, and a set of images (SetOfParticles, SetOfAverages, or SetOfClasses2D) for gray value adjustment. The outputs are processed particles or volumes, saved as new stacks or metadata files. For particles, the output includes normalized, centered, and phase-flipped images, with optional dust removal and thresholding. For volumes, the output includes symmetrized, denoised, and segmented maps, with optional contrast inversion and thresholding. The protocol generates intermediate files for steps like randomization, thresholding, and normalization, and final outputs are ready for further processing, such as 3D reconstruction or classification. The outputs are compatible with other Scipion protocols, ensuring seamless integration into larger workflows.'''

######SPLIT PHRASES
for key, value in dictProtocolSummary.items():
    dictProtocols[key] = dictProtocols.get(key, {})
    dictProtocols[key]['text'] = dictProtocols[key].get('text', {})
    dictProtocols[key]['text']['summary'] = value[:value.find(splittersSummary1)].split('.')
    dictProtocols[key]['text']['parameters'] = value[value.find(splittersSummary1):value.find(splittersSummary2)].split('.')
    dictProtocols[key]['text']['IO'] = value[value.find(splittersSummary2):].split('.')


######EMBEDDING PHRASES
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
userQuestionVector1 = embeddings.embed_query(userQuestion1)
userQuestionVector2 = embeddings.embed_query(userQuestion2)
userQuestionVector3 = embeddings.embed_query(userQuestion3)



for key in dictProtocolSummary.keys():
    dictProtocols[key]['vector'] = dictProtocols[key].get('vector', {})
    dictProtocols[key]['vector']['summary'] = []
    dictProtocols[key]['vector']['parameters'] = []
    dictProtocols[key]['vector']['IO'] = []
    for phrase in dictProtocols[key]['text']['summary']:
        dictProtocols[key]['vector']['summary'].append(embeddings.embed_query(phrase))
    for phrase in dictProtocols[key]['text']['parameters']:
        dictProtocols[key]['vector']['parameters'].append(embeddings.embed_query(phrase))
    for phrase in dictProtocols[key]['text']['IO']:
        dictProtocols[key]['vector']['IO'].append(embeddings.embed_query(phrase))


######CALCULATE DISTANCES CORRELLATION
import faiss #faiss-cpu
import numpy as np

dimension = len(userQuestionVector1)
dictIndex = {'summary': faiss.IndexFlatIP(dimension),
            'parameters':faiss.IndexFlatIP(dimension),
            'IO': faiss.IndexFlatIP(dimension)}

def normalize(vecs, do:bool = True):
    vecs = np.array(vecs)
    vecs = vecs.reshape(1, -1)
    if do:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_normalized = vecs / norms  # Normalizar el vector
        vecs =  vecs_normalized.flatten()
    return vecs
listProtocolsSummaryShorted = []
listProtocolsParametersShorted = []
listProtocolsIOShorted = []
for key in dictProtocolSummary.keys():

    for vect in dictProtocols[key]['vector']['summary']:
        vector = normalize(vect, do=False)  # Normalize for cosine similarity
        dictIndex['summary'].add(vector)
        listProtocolsSummaryShorted.append(key)
    for vect in dictProtocols[key]['vector']['parameters']:
        vector = normalize(vect, do=False)  # Normalize for cosine similarity
        dictIndex['parameters'].add(vector)
        listProtocolsParametersShorted.append(key)
    for vect in dictProtocols[key]['vector']['IO']:
        vector = normalize(vect, do=False)  # Normalize for cosine similarity
        dictIndex['IO'].add(vector)
        listProtocolsIOShorted.append(key)


dictResults = {'summary': {'Correlation': None, 'Index': None},
               'parameters': {'Correlation': None, 'Index': None},
               'IO': {'Correlation': None, 'Index': None}}

dictResults['summary']['Correlation'], dictResults['summary']['Index'] = dictIndex['summary'].search(normalize(userQuestionVector1, do=False), k=3)  # Search 5 nearest neighbors
dictResults['parameters']['Correlation'], dictResults['parameters']['Index'] =  dictIndex['parameters'].search(normalize(userQuestionVector1, do=False), k=3)  # Search 5 nearest neighbors
dictResults['IO']['Correlation'], dictResults['IO']['Index'] = dictIndex['IO'].search(normalize(userQuestionVector1, do=False), k=3)  # Search 5 nearest neighbors



######RESULTS
print(f'\n####\nQuestion: {userQuestion1}: ')
print(f"First similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,0])]}: {round(float(dictResults['summary']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,1])]}: {round(float(dictResults['summary']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,2])]}: {round(float(dictResults['summary']['Correlation'][0, 2]), 2)} correlation")
print(f"First similar  protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,0])]}: {round(float(dictResults['parameters']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,1])]}: {round(float(dictResults['parameters']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,2])]}: {round(float(dictResults['parameters']['Correlation'][0, 2]), 2)} correlation")
print(f"First similar  protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,0])]}: {round(float(dictResults['IO']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,1])]}: {round(float(dictResults['IO']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,2])]}: {round(float(dictResults['IO']['Correlation'][0, 2]), 2)} correlation")

dictResults['summary']['Correlation'], dictResults['summary']['Index'] = dictIndex['summary'].search(normalize(userQuestionVector2, do=False), k=3)
dictResults['parameters']['Correlation'], dictResults['parameters']['Index'] =  dictIndex['parameters'].search(normalize(userQuestionVector2, do=False), k=3)
dictResults['IO']['Correlation'], dictResults['IO']['Index'] = dictIndex['IO'].search(normalize(userQuestionVector2, do=False), k=3)

print(f'\n####\nQuestion: {userQuestion2}: ')
print(f"First similar  protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,0])]}: {round(float(dictResults['summary']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,1])]}: {round(float(dictResults['summary']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,2])]}: {round(float(dictResults['summary']['Correlation'][0, 2]), 2)} correlation")
print(f"First similar  protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,0])]}: {round(float(dictResults['parameters']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,1])]}: {round(float(dictResults['parameters']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,2])]}: {round(float(dictResults['parameters']['Correlation'][0, 2]), 2)} correlation")
print(f"First similar  protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,0])]}: {round(float(dictResults['IO']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,1])]}: {round(float(dictResults['IO']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,2])]}: {round(float(dictResults['IO']['Correlation'][0, 2]), 2)} correlation")
dictResults['summary']['Correlation'], dictResults['summary']['Index'] = dictIndex['summary'].search(normalize(userQuestionVector3, do=False), k=3)
dictResults['parameters']['Correlation'], dictResults['parameters']['Index'] =  dictIndex['parameters'].search(normalize(userQuestionVector3, do=False), k=3)
dictResults['IO']['Correlation'], dictResults['IO']['Index'] = dictIndex['IO'].search(normalize(userQuestionVector3, do=False), k=3)

print(f'\n####\nQuestion: {userQuestion3}: ')
print(f"First similar  protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,0])]}: {round(float(dictResults['summary']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,1])]}: {round(float(dictResults['summary']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in Summary: {listProtocolsSummaryShorted[int(dictResults['summary']['Index'][0,2])]}: {round(float(dictResults['summary']['Correlation'][0, 2]), 2)} correlation")
print(f"First similar  protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,0])]}: {round(float(dictResults['parameters']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,1])]}: {round(float(dictResults['parameters']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in parameters: {listProtocolsParametersShorted[int(dictResults['parameters']['Index'][0,2])]}: {round(float(dictResults['parameters']['Correlation'][0, 2]), 2)} correlation")
print(f"First similar  protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,0])]}: {round(float(dictResults['IO']['Correlation'][0, 0]), 2)} correlation")
print(f"Second similar protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,1])]}: {round(float(dictResults['IO']['Correlation'][0, 1]), 2)} correlation")
print(f"Third similar protocol in IO: {listProtocolsIOShorted[int(dictResults['IO']['Index'][0,2])]}: {round(float(dictResults['IO']['Correlation'][0, 2]), 2)} correlation")

######SCORE



######PRINT RESULTS
