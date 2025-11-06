import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';

// === 기본 설정 ===
const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// viewer 영역에 삽입
document.getElementById('viewer').appendChild(renderer.domElement);

// === 카메라 설정 ===
const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.4, 2.5);

// === 조명 ===
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(1, 1, 2);
scene.add(light);
scene.add(new THREE.AmbientLight(0xffffff, 0.4));

// === 애니메이팅 관련 변수 선언 === 
let vrm;
let allPoseData = []; // 모든 포즈 데이터를 저장할 배열
let currentAnim = 0;  // 현재 재생 중인 애니메이션(json) 인덱스
let currentFrame = 0; // 현재 프레임 인덱스


// === VRM 로드 ===
const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));

loader.load(
  'transwulator.vrm',
  gltf => {
    vrm = gltf.userData.vrm;
    scene.add(vrm.scene);
    vrm.scene.position.y = 0.4; // 건드리지 않기! 높이 맞춰둔거에요...
    loadAllPoseData(); // 모든 포즈 데이터 로드 함수 호출

    const bbox = new THREE.Box3().setFromObject(vrm.scene);
    const size = bbox.getSize(new THREE.Vector3()).length();
    const center = bbox.getCenter(new THREE.Vector3());

    camera.position.set(center.x, center.y + size * 0.1, center.z + size * 1.2);
    camera.lookAt(center);
  },
  xhr => console.log(`${(xhr.loaded / xhr.total * 100).toFixed(1)}% loaded`),
  err => console.error('VRM 로드 오류:', err)
);



// =============================================
// ================== 애니메이팅 관련 함수들 선언 ===
// =============================================

async function loadAllPoseData() {
  const folderPath = '/posedata/pose/';

  try {
    // ⚠️ 서버 환경에서 폴더 내 파일 목록을 직접 읽는 기능은 JS 단독으론 불가하므로,
    // 파일 목록 JSON (예: pose_index.json) 혹은 서버 라우팅을 통해 제공하는 방식을 권장합니다.
    // 만약 파일 이름이 미리 알 수 있다면 아래 fetch 부분을 그대로 사용하세요.

    // 👉 로컬에서 실행 중이라면 fetch 대신 수동으로 목록을 가져오는 helper 준비 필요.
    // 여기선 fetch로 목록을 불러오는 예시를 들어 설명합니다.
    const res = await fetch(folderPath);
    const text = await res.text();

    // 폴더 목록 HTML 파싱 (Apache/Nginx 디렉터리 인덱스 형태 기준)
    const fileMatches = [...text.matchAll(/href="([^"]+\.json)"/g)];
    const jsonFiles = fileMatches.map(m => m[1]);

    // 파일명에 포함된 숫자를 기준으로 정렬 (예: 001_pose.json → 1 → 오름차순)
    jsonFiles.sort((a, b) => {
      const numA = parseInt(a.match(/\d+/)?.[0] || '0');
      const numB = parseInt(b.match(/\d+/)?.[0] || '0');
      return numA - numB;
    });

    console.log(`📂 pose 폴더에서 ${jsonFiles.length}개의 파일 발견`);

    // 각 JSON 파일 로드
    for (const file of jsonFiles) {
      const path = `${folderPath}${file}`;
      try {
        const res = await fetch(path);
        const data = await res.json();
        allPoseData.push(data);
        console.log(`✅ 로드됨: ${file} (${data.length} frames)`);
      } catch (err) {
        console.error(`❌ ${file} 로드 실패:`, err);
      }
    }

  } catch (err) {
    console.error('❌ pose 폴더 접근 오류:', err);
  }
}


// === 좌표 → 회전 변환 ===
function calcQuat(a, b) {
  const dir = new THREE.Vector3(b[0]-a[0], b[1]-a[1], b[2]-a[2]).normalize();
  const base = new THREE.Vector3(0, -1, 0);
  return new THREE.Quaternion().setFromUnitVectors(base, dir);
}

// === 본 회전 적용 ===
function updatePose(vrm, pose, hands) {
  if (!pose?.length) return;

  const leftUpperArmQ = calcQuat(pose[11], pose[13]);
  const rightUpperArmQ = calcQuat(pose[12], pose[14]);
  const leftLowerArmQ = calcQuat(pose[13], pose[15]);
  const rightLowerArmQ = calcQuat(pose[14], pose[16]);

  vrm.humanoid.getBoneNode('leftUpperArm')?.quaternion.copy(leftUpperArmQ);
  vrm.humanoid.getBoneNode('rightUpperArm')?.quaternion.copy(rightUpperArmQ);
  vrm.humanoid.getBoneNode('leftLowerArm')?.quaternion.copy(leftLowerArmQ);
  vrm.humanoid.getBoneNode('rightLowerArm')?.quaternion.copy(rightLowerArmQ);

  // 손 회전
  if (hands?.[0]) {
    const wristQ = calcQuat(hands[0][0], hands[0][9]);
    vrm.humanoid.getBoneNode('leftHand')?.quaternion.slerp(wristQ, 0.3);
  }
  if (hands?.[1]) {
    const wristQ = calcQuat(hands[1][0], hands[1][9]);
    vrm.humanoid.getBoneNode('rightHand')?.quaternion.slerp(wristQ, 0.3);
  }
}
// ===============================================================
// ===============================================================
// ===================== 애니메이팅 관련 함수들 선언 =================
// ===============================================================
// ===============================================================

// === 리사이즈 대응 ===
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// === 렌더링 루프 ===
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
