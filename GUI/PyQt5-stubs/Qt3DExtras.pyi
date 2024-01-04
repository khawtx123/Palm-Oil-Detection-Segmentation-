# The PEP 484 type hints stub file for the Qt3DExtras module.
#
# Generated by SIP 6.4.0
#
# Copyright (c) 2021 Riverbank Computing Limited <info@riverbankcomputing.com>
# 
# This file is part of PyQt3D.
# 
# This file may be used under the terms of the GNU General Public License
# version 3.0 as published by the Free Software Foundation and appearing in
# the file LICENSE included in the packaging of this file.  Please review the
# following information to ensure the GNU General Public License version 3.0
# requirements will be met: http://www.gnu.org/copyleft/gpl.html.
# 
# If you do not wish to use this file under the terms of the GPL version 3.0
# then you may purchase a commercial license.  For more information contact
# info@riverbankcomputing.com.
# 
# This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
# WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.


import typing

from PyQt5 import sip
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import Qt3DRender
from PyQt5 import Qt3DInput
from PyQt5 import Qt3DCore

# Support for QDate, QDateTime and QTime.
import datetime

# Convenient type aliases.
PYQT_SLOT = typing.Union[typing.Callable[..., None], QtCore.pyqtBoundSignal]

# Convenient aliases for complicated OpenGL types.
PYQT_OPENGL_ARRAY = typing.Union[typing.Sequence[int], typing.Sequence[float],
        sip.Buffer, None]
PYQT_OPENGL_BOUND_ARRAY = typing.Union[typing.Sequence[int],
        typing.Sequence[float], sip.Buffer, int, None]


class QAbstractCameraController(Qt3DCore.QEntity):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    def mouseDevice(self) -> Qt3DInput.QMouseDevice: ...
    def keyboardDevice(self) -> Qt3DInput.QKeyboardDevice: ...
    decelerationChanged: typing.ClassVar[QtCore.pyqtSignal]
    accelerationChanged: typing.ClassVar[QtCore.pyqtSignal]
    lookSpeedChanged: typing.ClassVar[QtCore.pyqtSignal]
    linearSpeedChanged: typing.ClassVar[QtCore.pyqtSignal]
    cameraChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setDeceleration(self, deceleration: float) -> None: ...
    def setAcceleration(self, acceleration: float) -> None: ...
    def setLookSpeed(self, lookSpeed: float) -> None: ...
    def setLinearSpeed(self, linearSpeed: float) -> None: ...
    def setCamera(self, camera: Qt3DRender.QCamera) -> None: ...
    def deceleration(self) -> float: ...
    def acceleration(self) -> float: ...
    def lookSpeed(self) -> float: ...
    def linearSpeed(self) -> float: ...
    def camera(self) -> Qt3DRender.QCamera: ...

class QAbstractSpriteSheet(Qt3DCore.QNode):

    currentIndexChanged: typing.ClassVar[QtCore.pyqtSignal]
    textureTransformChanged: typing.ClassVar[QtCore.pyqtSignal]
    textureChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setCurrentIndex(self, currentIndex: int) -> None: ...
    def setTexture(self, texture: Qt3DRender.QAbstractTexture) -> None: ...
    def currentIndex(self) -> int: ...
    def textureTransform(self) -> QtGui.QMatrix3x3: ...
    def texture(self) -> Qt3DRender.QAbstractTexture: ...

class QConeGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    lengthChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    bottomRadiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    topRadiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    hasBottomEndcapChanged: typing.ClassVar[QtCore.pyqtSignal]
    hasTopEndcapChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setLength(self, length: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def setBottomRadius(self, bottomRadius: float) -> None: ...
    def setTopRadius(self, topRadius: float) -> None: ...
    def setHasBottomEndcap(self, hasBottomEndcap: bool) -> None: ...
    def setHasTopEndcap(self, hasTopEndcap: bool) -> None: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def texCoordAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...
    def length(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...
    def bottomRadius(self) -> float: ...
    def topRadius(self) -> float: ...
    def hasBottomEndcap(self) -> bool: ...
    def hasTopEndcap(self) -> bool: ...
    def updateIndices(self) -> None: ...
    def updateVertices(self) -> None: ...

class QConeMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    lengthChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    bottomRadiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    topRadiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    hasBottomEndcapChanged: typing.ClassVar[QtCore.pyqtSignal]
    hasTopEndcapChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setLength(self, length: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def setBottomRadius(self, bottomRadius: float) -> None: ...
    def setTopRadius(self, topRadius: float) -> None: ...
    def setHasBottomEndcap(self, hasBottomEndcap: bool) -> None: ...
    def setHasTopEndcap(self, hasTopEndcap: bool) -> None: ...
    def length(self) -> float: ...
    def bottomRadius(self) -> float: ...
    def topRadius(self) -> float: ...
    def hasBottomEndcap(self) -> bool: ...
    def hasTopEndcap(self) -> bool: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...

class QCuboidGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    xyMeshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    xzMeshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    yzMeshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    zExtentChanged: typing.ClassVar[QtCore.pyqtSignal]
    yExtentChanged: typing.ClassVar[QtCore.pyqtSignal]
    xExtentChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setXYMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setXZMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setYZMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setZExtent(self, zExtent: float) -> None: ...
    def setYExtent(self, yExtent: float) -> None: ...
    def setXExtent(self, xExtent: float) -> None: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def tangentAttribute(self) -> Qt3DRender.QAttribute: ...
    def texCoordAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...
    def xzMeshResolution(self) -> QtCore.QSize: ...
    def xyMeshResolution(self) -> QtCore.QSize: ...
    def yzMeshResolution(self) -> QtCore.QSize: ...
    def zExtent(self) -> float: ...
    def yExtent(self) -> float: ...
    def xExtent(self) -> float: ...
    def updateVertices(self) -> None: ...
    def updateIndices(self) -> None: ...

class QCuboidMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    xyMeshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    xzMeshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    yzMeshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    zExtentChanged: typing.ClassVar[QtCore.pyqtSignal]
    yExtentChanged: typing.ClassVar[QtCore.pyqtSignal]
    xExtentChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setXYMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setXZMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setYZMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setZExtent(self, zExtent: float) -> None: ...
    def setYExtent(self, yExtent: float) -> None: ...
    def setXExtent(self, xExtent: float) -> None: ...
    def xyMeshResolution(self) -> QtCore.QSize: ...
    def xzMeshResolution(self) -> QtCore.QSize: ...
    def yzMeshResolution(self) -> QtCore.QSize: ...
    def zExtent(self) -> float: ...
    def yExtent(self) -> float: ...
    def xExtent(self) -> float: ...

class QCylinderGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    lengthChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    radiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setLength(self, length: float) -> None: ...
    def setRadius(self, radius: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def texCoordAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...
    def length(self) -> float: ...
    def radius(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...
    def updateIndices(self) -> None: ...
    def updateVertices(self) -> None: ...

class QCylinderMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    lengthChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    radiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setLength(self, length: float) -> None: ...
    def setRadius(self, radius: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def length(self) -> float: ...
    def radius(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...

class QDiffuseMapMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    textureScaleChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setTextureScale(self, textureScale: float) -> None: ...
    def setDiffuse(self, diffuse: Qt3DRender.QAbstractTexture) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setAmbient(self, color: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def textureScale(self) -> float: ...
    def diffuse(self) -> Qt3DRender.QAbstractTexture: ...
    def shininess(self) -> float: ...
    def specular(self) -> QtGui.QColor: ...
    def ambient(self) -> QtGui.QColor: ...

class QDiffuseSpecularMapMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    textureScaleChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setTextureScale(self, textureScale: float) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: Qt3DRender.QAbstractTexture) -> None: ...
    def setDiffuse(self, diffuse: Qt3DRender.QAbstractTexture) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def textureScale(self) -> float: ...
    def shininess(self) -> float: ...
    def specular(self) -> Qt3DRender.QAbstractTexture: ...
    def diffuse(self) -> Qt3DRender.QAbstractTexture: ...
    def ambient(self) -> QtGui.QColor: ...

class QDiffuseSpecularMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    alphaBlendingEnabledChanged: typing.ClassVar[QtCore.pyqtSignal]
    textureScaleChanged: typing.ClassVar[QtCore.pyqtSignal]
    normalChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setAlphaBlendingEnabled(self, enabled: bool) -> None: ...
    def setTextureScale(self, textureScale: float) -> None: ...
    def setNormal(self, normal: typing.Any) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: typing.Any) -> None: ...
    def setDiffuse(self, diffuse: typing.Any) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def isAlphaBlendingEnabled(self) -> bool: ...
    def textureScale(self) -> float: ...
    def normal(self) -> typing.Any: ...
    def shininess(self) -> float: ...
    def specular(self) -> typing.Any: ...
    def diffuse(self) -> typing.Any: ...
    def ambient(self) -> QtGui.QColor: ...

class QExtrudedTextGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    depthChanged: typing.ClassVar[QtCore.pyqtSignal]
    fontChanged: typing.ClassVar[QtCore.pyqtSignal]
    textChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setDepth(self, extrusionLength: float) -> None: ...
    def setFont(self, font: QtGui.QFont) -> None: ...
    def setText(self, text: str) -> None: ...
    def extrusionLength(self) -> float: ...
    def font(self) -> QtGui.QFont: ...
    def text(self) -> str: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...

class QExtrudedTextMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    depthChanged: typing.ClassVar[QtCore.pyqtSignal]
    fontChanged: typing.ClassVar[QtCore.pyqtSignal]
    textChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setDepth(self, depth: float) -> None: ...
    def setFont(self, font: QtGui.QFont) -> None: ...
    def setText(self, text: str) -> None: ...
    def depth(self) -> float: ...
    def font(self) -> QtGui.QFont: ...
    def text(self) -> str: ...

class QFirstPersonCameraController('QAbstractCameraController'):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

class QForwardRenderer(Qt3DRender.QTechniqueFilter):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    showDebugOverlayChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setShowDebugOverlay(self, showDebugOverlay: bool) -> None: ...
    def showDebugOverlay(self) -> bool: ...
    buffersToClearChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setBuffersToClear(self, a0: Qt3DRender.QClearBuffers.BufferType) -> None: ...
    def buffersToClear(self) -> Qt3DRender.QClearBuffers.BufferType: ...
    gammaChanged: typing.ClassVar[QtCore.pyqtSignal]
    frustumCullingEnabledChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setGamma(self, gamma: float) -> None: ...
    def setFrustumCullingEnabled(self, enabled: bool) -> None: ...
    def gamma(self) -> float: ...
    def isFrustumCullingEnabled(self) -> bool: ...
    externalRenderTargetSizeChanged: typing.ClassVar[QtCore.pyqtSignal]
    surfaceChanged: typing.ClassVar[QtCore.pyqtSignal]
    cameraChanged: typing.ClassVar[QtCore.pyqtSignal]
    clearColorChanged: typing.ClassVar[QtCore.pyqtSignal]
    viewportRectChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setExternalRenderTargetSize(self, size: QtCore.QSize) -> None: ...
    def setSurface(self, surface: QtCore.QObject) -> None: ...
    def setCamera(self, camera: Qt3DCore.QEntity) -> None: ...
    def setClearColor(self, clearColor: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setViewportRect(self, viewportRect: QtCore.QRectF) -> None: ...
    def externalRenderTargetSize(self) -> QtCore.QSize: ...
    def surface(self) -> QtCore.QObject: ...
    def camera(self) -> Qt3DCore.QEntity: ...
    def clearColor(self) -> QtGui.QColor: ...
    def viewportRect(self) -> QtCore.QRectF: ...

class QGoochMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    betaChanged: typing.ClassVar[QtCore.pyqtSignal]
    alphaChanged: typing.ClassVar[QtCore.pyqtSignal]
    warmChanged: typing.ClassVar[QtCore.pyqtSignal]
    coolChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setShininess(self, shininess: float) -> None: ...
    def setBeta(self, beta: float) -> None: ...
    def setAlpha(self, alpha: float) -> None: ...
    def setWarm(self, warm: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setCool(self, cool: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setSpecular(self, specular: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setDiffuse(self, diffuse: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def shininess(self) -> float: ...
    def beta(self) -> float: ...
    def alpha(self) -> float: ...
    def warm(self) -> QtGui.QColor: ...
    def cool(self) -> QtGui.QColor: ...
    def specular(self) -> QtGui.QColor: ...
    def diffuse(self) -> QtGui.QColor: ...

class QMetalRoughMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    textureScaleChanged: typing.ClassVar[QtCore.pyqtSignal]
    normalChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientOcclusionChanged: typing.ClassVar[QtCore.pyqtSignal]
    roughnessChanged: typing.ClassVar[QtCore.pyqtSignal]
    metalnessChanged: typing.ClassVar[QtCore.pyqtSignal]
    baseColorChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setTextureScale(self, textureScale: float) -> None: ...
    def setNormal(self, normal: typing.Any) -> None: ...
    def setAmbientOcclusion(self, ambientOcclusion: typing.Any) -> None: ...
    def setRoughness(self, roughness: typing.Any) -> None: ...
    def setMetalness(self, metalness: typing.Any) -> None: ...
    def setBaseColor(self, baseColor: typing.Any) -> None: ...
    def textureScale(self) -> float: ...
    def normal(self) -> typing.Any: ...
    def ambientOcclusion(self) -> typing.Any: ...
    def roughness(self) -> typing.Any: ...
    def metalness(self) -> typing.Any: ...
    def baseColor(self) -> typing.Any: ...

class QMorphPhongMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    interpolatorChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setInterpolator(self, interpolator: float) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setDiffuse(self, diffuse: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def interpolator(self) -> float: ...
    def shininess(self) -> float: ...
    def specular(self) -> QtGui.QColor: ...
    def diffuse(self) -> QtGui.QColor: ...
    def ambient(self) -> QtGui.QColor: ...

class QNormalDiffuseMapMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    textureScaleChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    normalChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setTextureScale(self, textureScale: float) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setNormal(self, normal: Qt3DRender.QAbstractTexture) -> None: ...
    def setDiffuse(self, diffuse: Qt3DRender.QAbstractTexture) -> None: ...
    def setSpecular(self, specular: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def textureScale(self) -> float: ...
    def shininess(self) -> float: ...
    def normal(self) -> Qt3DRender.QAbstractTexture: ...
    def diffuse(self) -> Qt3DRender.QAbstractTexture: ...
    def specular(self) -> QtGui.QColor: ...
    def ambient(self) -> QtGui.QColor: ...

class QNormalDiffuseMapAlphaMaterial('QNormalDiffuseMapMaterial'):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

class QNormalDiffuseSpecularMapMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    textureScaleChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    normalChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setTextureScale(self, textureScale: float) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: Qt3DRender.QAbstractTexture) -> None: ...
    def setNormal(self, normal: Qt3DRender.QAbstractTexture) -> None: ...
    def setDiffuse(self, diffuse: Qt3DRender.QAbstractTexture) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def textureScale(self) -> float: ...
    def shininess(self) -> float: ...
    def specular(self) -> Qt3DRender.QAbstractTexture: ...
    def normal(self) -> Qt3DRender.QAbstractTexture: ...
    def diffuse(self) -> Qt3DRender.QAbstractTexture: ...
    def ambient(self) -> QtGui.QColor: ...

class QOrbitCameraController('QAbstractCameraController'):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    zoomInLimitChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setZoomInLimit(self, zoomInLimit: float) -> None: ...
    def zoomInLimit(self) -> float: ...

class QPerVertexColorMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

class QPhongAlphaMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    blendFunctionArgChanged: typing.ClassVar[QtCore.pyqtSignal]
    destinationAlphaArgChanged: typing.ClassVar[QtCore.pyqtSignal]
    sourceAlphaArgChanged: typing.ClassVar[QtCore.pyqtSignal]
    destinationRgbArgChanged: typing.ClassVar[QtCore.pyqtSignal]
    sourceRgbArgChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setBlendFunctionArg(self, blendFunctionArg: Qt3DRender.QBlendEquation.BlendFunction) -> None: ...
    def setDestinationAlphaArg(self, destinationAlphaArg: Qt3DRender.QBlendEquationArguments.Blending) -> None: ...
    def setSourceAlphaArg(self, sourceAlphaArg: Qt3DRender.QBlendEquationArguments.Blending) -> None: ...
    def setDestinationRgbArg(self, destinationRgbArg: Qt3DRender.QBlendEquationArguments.Blending) -> None: ...
    def setSourceRgbArg(self, sourceRgbArg: Qt3DRender.QBlendEquationArguments.Blending) -> None: ...
    def blendFunctionArg(self) -> Qt3DRender.QBlendEquation.BlendFunction: ...
    def destinationAlphaArg(self) -> Qt3DRender.QBlendEquationArguments.Blending: ...
    def sourceAlphaArg(self) -> Qt3DRender.QBlendEquationArguments.Blending: ...
    def destinationRgbArg(self) -> Qt3DRender.QBlendEquationArguments.Blending: ...
    def sourceRgbArg(self) -> Qt3DRender.QBlendEquationArguments.Blending: ...
    alphaChanged: typing.ClassVar[QtCore.pyqtSignal]
    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setAlpha(self, alpha: float) -> None: ...
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setDiffuse(self, diffuse: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def alpha(self) -> float: ...
    def shininess(self) -> float: ...
    def specular(self) -> QtGui.QColor: ...
    def diffuse(self) -> QtGui.QColor: ...
    def ambient(self) -> QtGui.QColor: ...

class QPhongMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    shininessChanged: typing.ClassVar[QtCore.pyqtSignal]
    specularChanged: typing.ClassVar[QtCore.pyqtSignal]
    diffuseChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setShininess(self, shininess: float) -> None: ...
    def setSpecular(self, specular: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setDiffuse(self, diffuse: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def setAmbient(self, ambient: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def shininess(self) -> float: ...
    def specular(self) -> QtGui.QColor: ...
    def diffuse(self) -> QtGui.QColor: ...
    def ambient(self) -> QtGui.QColor: ...

class QPlaneGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    mirroredChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setMirrored(self, mirrored: bool) -> None: ...
    def mirrored(self) -> bool: ...
    heightChanged: typing.ClassVar[QtCore.pyqtSignal]
    widthChanged: typing.ClassVar[QtCore.pyqtSignal]
    resolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setHeight(self, height: float) -> None: ...
    def setWidth(self, width: float) -> None: ...
    def setResolution(self, resolution: QtCore.QSize) -> None: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def tangentAttribute(self) -> Qt3DRender.QAttribute: ...
    def texCoordAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...
    def height(self) -> float: ...
    def width(self) -> float: ...
    def resolution(self) -> QtCore.QSize: ...
    def updateIndices(self) -> None: ...
    def updateVertices(self) -> None: ...

class QPlaneMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    mirroredChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setMirrored(self, mirrored: bool) -> None: ...
    def mirrored(self) -> bool: ...
    heightChanged: typing.ClassVar[QtCore.pyqtSignal]
    widthChanged: typing.ClassVar[QtCore.pyqtSignal]
    meshResolutionChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setMeshResolution(self, resolution: QtCore.QSize) -> None: ...
    def setHeight(self, height: float) -> None: ...
    def setWidth(self, width: float) -> None: ...
    def meshResolution(self) -> QtCore.QSize: ...
    def height(self) -> float: ...
    def width(self) -> float: ...

class QSkyboxEntity(Qt3DCore.QEntity):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    gammaCorrectEnabledChanged: typing.ClassVar[QtCore.pyqtSignal]
    baseNameChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setGammaCorrectEnabled(self, enabled: bool) -> None: ...
    def isGammaCorrectEnabled(self) -> bool: ...
    extensionChanged: typing.ClassVar[QtCore.pyqtSignal]
    def extension(self) -> str: ...
    def setExtension(self, extension: str) -> None: ...
    def baseName(self) -> str: ...
    def setBaseName(self, path: str) -> None: ...

class QSphereGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    generateTangentsChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    radiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setGenerateTangents(self, gen: bool) -> None: ...
    def setRadius(self, radius: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def tangentAttribute(self) -> Qt3DRender.QAttribute: ...
    def texCoordAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...
    def radius(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...
    def generateTangents(self) -> bool: ...
    def updateIndices(self) -> None: ...
    def updateVertices(self) -> None: ...

class QSphereMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    generateTangentsChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    radiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setGenerateTangents(self, gen: bool) -> None: ...
    def setRadius(self, radius: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def generateTangents(self) -> bool: ...
    def radius(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...

class QSpriteGrid('QAbstractSpriteSheet'):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    columnsChanged: typing.ClassVar[QtCore.pyqtSignal]
    rowsChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setColumns(self, columns: int) -> None: ...
    def setRows(self, rows: int) -> None: ...
    def columns(self) -> int: ...
    def rows(self) -> int: ...

class QSpriteSheet('QAbstractSpriteSheet'):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    spritesChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setSprites(self, sprites: typing.Iterable['QSpriteSheetItem']) -> None: ...
    def removeSprite(self, sprite: 'QSpriteSheetItem') -> None: ...
    @typing.overload
    def addSprite(self, x: int, y: int, width: int, height: int) -> 'QSpriteSheetItem': ...
    @typing.overload
    def addSprite(self, sprite: 'QSpriteSheetItem') -> None: ...
    def sprites(self) -> typing.List['QSpriteSheetItem']: ...

class QSpriteSheetItem(Qt3DCore.QNode):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    heightChanged: typing.ClassVar[QtCore.pyqtSignal]
    widthChanged: typing.ClassVar[QtCore.pyqtSignal]
    yChanged: typing.ClassVar[QtCore.pyqtSignal]
    xChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setHeight(self, height: int) -> None: ...
    def setWidth(self, width: int) -> None: ...
    def setY(self, y: int) -> None: ...
    def setX(self, x: int) -> None: ...
    def height(self) -> int: ...
    def width(self) -> int: ...
    def y(self) -> int: ...
    def x(self) -> int: ...

class Qt3DWindow(QtGui.QWindow):

    def __init__(self, screen: typing.Optional[QtGui.QScreen] = ...) -> None: ...

    def event(self, e: QtCore.QEvent) -> bool: ...
    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None: ...
    def showEvent(self, e: QtGui.QShowEvent) -> None: ...
    def renderSettings(self) -> Qt3DRender.QRenderSettings: ...
    def camera(self) -> Qt3DRender.QCamera: ...
    def defaultFrameGraph(self) -> 'QForwardRenderer': ...
    def activeFrameGraph(self) -> Qt3DRender.QFrameGraphNode: ...
    def setActiveFrameGraph(self, activeFrameGraph: Qt3DRender.QFrameGraphNode) -> None: ...
    def setRootEntity(self, root: Qt3DCore.QEntity) -> None: ...
    @typing.overload
    def registerAspect(self, aspect: Qt3DCore.QAbstractAspect) -> None: ...
    @typing.overload
    def registerAspect(self, name: str) -> None: ...

class QText2DEntity(Qt3DCore.QEntity):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    heightChanged: typing.ClassVar[QtCore.pyqtSignal]
    widthChanged: typing.ClassVar[QtCore.pyqtSignal]
    textChanged: typing.ClassVar[QtCore.pyqtSignal]
    colorChanged: typing.ClassVar[QtCore.pyqtSignal]
    fontChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setHeight(self, height: float) -> None: ...
    def setWidth(self, width: float) -> None: ...
    def height(self) -> float: ...
    def width(self) -> float: ...
    def setText(self, text: str) -> None: ...
    def text(self) -> str: ...
    def setColor(self, color: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor]) -> None: ...
    def color(self) -> QtGui.QColor: ...
    def setFont(self, font: QtGui.QFont) -> None: ...
    def font(self) -> QtGui.QFont: ...

class QTexturedMetalRoughMaterial('QMetalRoughMaterial'):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    normalChanged: typing.ClassVar[QtCore.pyqtSignal]
    ambientOcclusionChanged: typing.ClassVar[QtCore.pyqtSignal]

class QTextureMaterial(Qt3DRender.QMaterial):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    alphaBlendingEnabledChanged: typing.ClassVar[QtCore.pyqtSignal]
    textureTransformChanged: typing.ClassVar[QtCore.pyqtSignal]
    textureOffsetChanged: typing.ClassVar[QtCore.pyqtSignal]
    textureChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setAlphaBlendingEnabled(self, enabled: bool) -> None: ...
    def setTextureTransform(self, matrix: QtGui.QMatrix3x3) -> None: ...
    def setTextureOffset(self, textureOffset: QtGui.QVector2D) -> None: ...
    def setTexture(self, texture: Qt3DRender.QAbstractTexture) -> None: ...
    def isAlphaBlendingEnabled(self) -> bool: ...
    def textureTransform(self) -> QtGui.QMatrix3x3: ...
    def textureOffset(self) -> QtGui.QVector2D: ...
    def texture(self) -> Qt3DRender.QAbstractTexture: ...

class QTorusGeometry(Qt3DRender.QGeometry):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    minorRadiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    radiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setMinorRadius(self, minorRadius: float) -> None: ...
    def setRadius(self, radius: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def indexAttribute(self) -> Qt3DRender.QAttribute: ...
    def texCoordAttribute(self) -> Qt3DRender.QAttribute: ...
    def normalAttribute(self) -> Qt3DRender.QAttribute: ...
    def positionAttribute(self) -> Qt3DRender.QAttribute: ...
    def minorRadius(self) -> float: ...
    def radius(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...
    def updateIndices(self) -> None: ...
    def updateVertices(self) -> None: ...

class QTorusMesh(Qt3DRender.QGeometryRenderer):

    def __init__(self, parent: typing.Optional[Qt3DCore.QNode] = ...) -> None: ...

    minorRadiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    slicesChanged: typing.ClassVar[QtCore.pyqtSignal]
    ringsChanged: typing.ClassVar[QtCore.pyqtSignal]
    radiusChanged: typing.ClassVar[QtCore.pyqtSignal]
    def setMinorRadius(self, minorRadius: float) -> None: ...
    def setRadius(self, radius: float) -> None: ...
    def setSlices(self, slices: int) -> None: ...
    def setRings(self, rings: int) -> None: ...
    def minorRadius(self) -> float: ...
    def radius(self) -> float: ...
    def slices(self) -> int: ...
    def rings(self) -> int: ...