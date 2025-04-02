#!/usr/bin/env python3
"""
RapidShot DirectX Diagnostic Tool
Helps identify DirectX initialization issues with RapidShot

This script tests each component of the DirectX initialization chain to help
identify the source of issues in the RapidShot library.
"""
import os
import sys
import ctypes
import platform
import traceback
from ctypes import wintypes
import comtypes
import struct
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rapidshot_diagnostics.log', mode='w')
    ]
)
logger = logging.getLogger('rapidshot_diagnostics')

# ANSI color codes for better terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

# Results tracking
results = {
    "pass": [],
    "warn": [],
    "fail": []
}

def print_header(title):
    """Print a formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 20} {title} {'=' * 20}{Colors.RESET}")

def print_result(status, message):
    """Print a result with appropriate formatting and track it"""
    if status.lower() == "pass":
        print(f"{Colors.GREEN}✓ PASS:{Colors.RESET} {message}")
        results["pass"].append(message)
    elif status.lower() == "warn":
        print(f"{Colors.YELLOW}⚠ WARNING:{Colors.RESET} {message}")
        results["warn"].append(message)
    elif status.lower() == "fail":
        print(f"{Colors.RED}✗ FAIL:{Colors.RESET} {message}")
        results["fail"].append(message)

def print_summary():
    """Print a summary of all test results"""
    print_header("TEST SUMMARY")
    
    print(f"{Colors.GREEN}PASSED:{Colors.RESET} {len(results['pass'])}")
    print(f"{Colors.YELLOW}WARNINGS:{Colors.RESET} {len(results['warn'])}")
    print(f"{Colors.RED}FAILED:{Colors.RESET} {len(results['fail'])}")
    
    if results["fail"]:
        print("\n" + Colors.RED + Colors.BOLD + "FAILED TESTS:" + Colors.RESET)
        for fail in results["fail"]:
            print(f"  • {fail}")
    
    if results["warn"]:
        print("\n" + Colors.YELLOW + Colors.BOLD + "WARNINGS:" + Colors.RESET)
        for warn in results["warn"]:
            print(f"  • {warn}")

    if not results["fail"]:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All critical tests passed!{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed. Please check the issues above.{Colors.RESET}")

def test_system_info():
    """Test system information to identify potential compatibility issues"""
    print_header("SYSTEM INFORMATION")
    
    try:
        print(f"{Colors.CYAN}OS:{Colors.RESET} {platform.system()} {platform.release()} {platform.version()}")
        print(f"{Colors.CYAN}Python:{Colors.RESET} {platform.python_version()}")
        print(f"{Colors.CYAN}Architecture:{Colors.RESET} {platform.architecture()[0]}")
        print(f"{Colors.CYAN}Processor:{Colors.RESET} {platform.processor()}")
        
        # Check Windows version compatibility
        if platform.system() == "Windows":
            if float(platform.release()) < 10:
                print_result("warn", f"Windows {platform.release()} might have limited DirectX support. Windows 10 or higher is recommended.")
            else:
                print_result("pass", f"Windows {platform.release()} should have good DirectX support")
                
        # Check Python architecture
        if platform.architecture()[0] == "32bit":
            print_result("warn", "Using 32-bit Python. 64-bit Python is recommended for best compatibility.")
        else:
            print_result("pass", "Using 64-bit Python")
            
        # Check processor architecture
        if "64" in platform.architecture()[0] and "32" in platform.processor():
            print_result("warn", "Running 64-bit Python on a 32-bit processor may cause issues")
            
        return True
    except Exception as e:
        print_result("fail", f"Error checking system information: {e}")
        return False

def test_directx_dlls():
    """Test DirectX DLLs availability and version information"""
    print_header("DIRECTX DLL VERIFICATION")
    
    # Check for Direct3D DLLs and their versions
    dll_checks = []
    dx_dlls = ["d3d11.dll", "dxgi.dll", "d3dcompiler_47.dll"]
    
    # Track overall status
    all_essential_dlls_available = True
    
    for dll in dx_dlls:
        try:
            # Try to load the DLL directly
            handle = ctypes.windll.LoadLibrary(dll)
            print(f"{Colors.GREEN}✓{Colors.RESET} {dll} is available")
            dll_checks.append(True)
            
            # Try to get file version info
            try:
                from ctypes import wintypes
                GetFileVersionInfoSizeW = ctypes.windll.version.GetFileVersionInfoSizeW
                GetFileVersionInfoW = ctypes.windll.version.GetFileVersionInfoW
                VerQueryValueW = ctypes.windll.version.VerQueryValueW
                
                # Get system directory where the DLL should be
                GetSystemDirectoryW = ctypes.windll.kernel32.GetSystemDirectoryW
                buffer = ctypes.create_unicode_buffer(260)
                GetSystemDirectoryW(buffer, len(buffer))
                
                # Full path to the DLL
                dll_path = os.path.join(buffer.value, dll)
                
                # Get version info size
                size = GetFileVersionInfoSizeW(dll_path, None)
                if size > 0:
                    data = (ctypes.c_ubyte * size)()
                    res = GetFileVersionInfoW(dll_path, 0, size, data)
                    if res:
                        info = ctypes.c_void_p()
                        length = wintypes.UINT()
                        res = VerQueryValueW(data, r"\VarFileInfo\Translation", 
                                           ctypes.byref(info), ctypes.byref(length))
                        if res:
                            # Version info structure
                            struct_fmt = "hhhh"
                            info_size = struct.calcsize(struct_fmt)
                            info_data = ctypes.string_at(info.value, info_size)
                            lang, codepage = struct.unpack(struct_fmt, info_data)[:2]
                            
                            version_path = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\FileVersion"
                            res = VerQueryValueW(data, version_path,
                                              ctypes.byref(info), ctypes.byref(length))
                            if res:
                                version_str = ctypes.wstring_at(info.value, length.value - 1)
                                print(f"  Version: {version_str}")
            except Exception as ve:
                print(f"  {Colors.YELLOW}Version information not available: {ve}{Colors.RESET}")
                
        except Exception as e:
            dll_checks.append(False)
            
            # Only count d3d11.dll and dxgi.dll as critical
            if dll in ["d3d11.dll", "dxgi.dll"]:
                all_essential_dlls_available = False
                print_result("fail", f"{dll} is not available: {e}")
            else:
                print_result("warn", f"{dll} is not available: {e}")
    
    # Check overall status
    if all_essential_dlls_available:
        print_result("pass", "All essential DirectX DLLs are available")
    else:
        print_result("fail", "Some essential DirectX DLLs are missing")
    
    return all_essential_dlls_available

def test_gpu_info():
    """Test GPU information using WMI"""
    print_header("GPU INFORMATION")
    
    has_directx_gpu = False
    
    # Check for GPU info using WMI
    try:
        import wmi
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            print(f"{Colors.CYAN}GPU:{Colors.RESET} {gpu.Name}")
            print(f"  Driver Version: {gpu.DriverVersion}")
            print(f"  Driver Date: {gpu.DriverDate}")
            print(f"  Video Mode: {gpu.VideoModeDescription}")
            print(f"  Adapter RAM: {int(gpu.AdapterRAM if hasattr(gpu, 'AdapterRAM') and gpu.AdapterRAM else 0)/1024/1024:.2f} MB")
            
            # Check for DirectX support
            if hasattr(gpu, "AdapterCompatibility"):
                print(f"  Adapter Compatibility: {gpu.AdapterCompatibility}")
                
                # Check if this is an NVIDIA, AMD or Intel GPU
                if "NVIDIA" in gpu.Name or "AMD" in gpu.Name or "ATI" in gpu.Name or "Intel" in gpu.Name:
                    print_result("pass", f"Found compatible GPU: {gpu.Name}")
                    has_directx_gpu = True
                else:
                    print_result("warn", f"Unknown GPU vendor: {gpu.Name}. Compatibility may vary.")
            
        if not has_directx_gpu:
            print_result("warn", "No known compatible GPU detected. DirectX support may be limited.")
        
        return True
        
    except ImportError:
        print_result("warn", "WMI module not available. Install with 'pip install wmi' for detailed GPU information.")
        
        # Fall back to dxdiag output
        try:
            import subprocess
            import tempfile
            
            # Create a temporary file for dxdiag output
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
                temp_path = temp.name
            
            # Run dxdiag and redirect output to the file
            print("Running dxdiag to gather GPU information (this may take a moment)...")
            subprocess.run(["dxdiag", "/t", temp_path], check=True)
            
            # Wait for file to be written
            time.sleep(2)
            
            # Read the file
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            # Parse for GPU info
            found_display = False
            gpu_found = False
            for line in lines:
                if "Display Devices" in line:
                    found_display = True
                    print(f"{Colors.CYAN}Display Devices:{Colors.RESET}")
                
                if found_display and ("Card name" in line or "Driver Version" in line or 
                                     "Driver Date" in line or "Display Memory" in line):
                    print(f"  {line.strip()}")
                    if "Card name" in line:
                        card_name = line.split(":")[1].strip() if ":" in line else "Unknown"
                        if "NVIDIA" in card_name or "AMD" in card_name or "ATI" in card_name or "Intel" in card_name:
                            print_result("pass", f"Found compatible GPU: {card_name}")
                            gpu_found = True
            
            if not gpu_found:
                print_result("warn", "No compatible GPU detected in dxdiag output.")
                
            # Clean up
            os.unlink(temp_path)
            
            return gpu_found
            
        except Exception as e:
            print_result("fail", f"Could not retrieve GPU information: {e}")
            return False
    except Exception as e:
        print_result("fail", f"Error retrieving GPU information: {e}")
        return False

# Begin DXGI/D3D11 Interface definitions for testing
# These are minimal definitions needed for testing
class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", wintypes.ULARGE_INTEGER),
        ("DedicatedSystemMemory", wintypes.ULARGE_INTEGER),
        ("SharedSystemMemory", wintypes.ULARGE_INTEGER),
        ("AdapterLuid", ctypes.c_longlong),
        ("Flags", wintypes.UINT),
    ]

def test_dxgi_factory_creation():
    """Test DXGI factory creation to verify basics of DirectX initialization"""
    print_header("DXGI FACTORY CREATION")
    
    # Try to create DXGI factory with progressively newer versions
    factory_versions = [
        {"name": "DXGI 1.6 (CreateDXGIFactory6)", "function": "CreateDXGIFactory6", "guid": "{c1b6694f-ff09-44a9-b03c-77900a0a1d17}"},
        {"name": "DXGI 1.4 (CreateDXGIFactory2)", "function": "CreateDXGIFactory2", "guid": "{1bc6ea02-ef36-464f-bf0c-21ca39e5168a}"},
        {"name": "DXGI 1.1 (CreateDXGIFactory1)", "function": "CreateDXGIFactory1", "guid": "{770aae78-f26f-4dba-a829-253c83d1b387}"},
        {"name": "DXGI 1.0 (CreateDXGIFactory)", "function": "CreateDXGIFactory", "guid": "{7b7166ec-21c7-44ae-b21a-c9ae321ae369}"},
    ]
    
    factory_created = False
    factory_handle = None
    factory_version = None
    
    for version in factory_versions:
        try:
            print(f"Trying {version['name']}...")
            create_factory = getattr(ctypes.windll.dxgi, version["function"], None)
            
            if create_factory is None:
                print_result("warn", f"{version['function']} not found in dxgi.dll")
                continue
                
            create_factory.restype = comtypes.HRESULT
            create_factory.argtypes = [
                ctypes.POINTER(comtypes.GUID),
                ctypes.POINTER(ctypes.c_void_p)
            ]
            
            factory_ptr = ctypes.c_void_p(0)
            guid = comtypes.GUID(version["guid"])
            
            hr = create_factory(ctypes.byref(guid), ctypes.byref(factory_ptr))
            
            if hr == 0 and factory_ptr.value:  # S_OK
                print_result("pass", f"Successfully created {version['name']}")
                factory_created = True
                factory_handle = factory_ptr
                factory_version = version
                break
            else:
                print_result("fail", f"Failed to create {version['name']}, HRESULT: {hr:#x}")
        except Exception as e:
            print_result("fail", f"Error trying {version['name']}: {e}")
    
    if not factory_created:
        print_result("fail", "Failed to create any DXGI factory")
        return False
    
    # If we have a factory, try to enumerate adapters
    adapters_found = 0
    
    # Factory interface types and GUIDs from lowest version that worked
    if factory_version["function"] == "CreateDXGIFactory6":
        factory_interface_guid = "{c1b6694f-ff09-44a9-b03c-77900a0a1d17}"  # IDXGIFactory6
    elif factory_version["function"] == "CreateDXGIFactory2":
        factory_interface_guid = "{1bc6ea02-ef36-464f-bf0c-21ca39e5168a}"  # IDXGIFactory4
    elif factory_version["function"] == "CreateDXGIFactory1":
        factory_interface_guid = "{770aae78-f26f-4dba-a829-253c83d1b387}"  # IDXGIFactory1
    else:
        factory_interface_guid = "{7b7166ec-21c7-44ae-b21a-c9ae321ae369}"  # IDXGIFactory
    
    # Define IDXGIAdapter1 vTable structure for minimal interface
    class IDXGIAdapter1_vTable(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            # IDXGIObject methods
            ("SetPrivateData", ctypes.c_void_p),
            ("SetPrivateDataInterface", ctypes.c_void_p),
            ("GetPrivateData", ctypes.c_void_p),
            ("GetParent", ctypes.c_void_p),
            # IDXGIAdapter methods
            ("EnumOutputs", ctypes.c_void_p),
            ("GetDesc", ctypes.c_void_p),
            ("CheckInterfaceSupport", ctypes.c_void_p),
            # IDXGIAdapter1 methods
            ("GetDesc1", ctypes.c_void_p),
        ]
    
    class IDXGIAdapter1(ctypes.Structure):
        _fields_ = [("lpVtbl", ctypes.POINTER(IDXGIAdapter1_vTable))]
    
    class IDXGIFactory1_vTable(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            # IDXGIObject methods
            ("SetPrivateData", ctypes.c_void_p),
            ("SetPrivateDataInterface", ctypes.c_void_p),
            ("GetPrivateData", ctypes.c_void_p),
            ("GetParent", ctypes.c_void_p),
            # IDXGIFactory methods
            ("EnumAdapters", ctypes.c_void_p),
            ("MakeWindowAssociation", ctypes.c_void_p),
            ("GetWindowAssociation", ctypes.c_void_p),
            ("CreateSwapChain", ctypes.c_void_p),
            ("CreateSoftwareAdapter", ctypes.c_void_p),
            # IDXGIFactory1 methods
            ("EnumAdapters1", ctypes.c_void_p),
        ]
    
    class IDXGIFactory1(ctypes.Structure):
        _fields_ = [("lpVtbl", ctypes.POINTER(IDXGIFactory1_vTable))]
    
    try:
        # Cast factory handle to IDXGIFactory1
        factory = ctypes.cast(factory_handle, ctypes.POINTER(IDXGIFactory1))
        
        # Use EnumAdapters1 to get adapters
        i = 0
        while True:
            try:
                adapter = ctypes.POINTER(IDXGIAdapter1)()
                
                # Extract the EnumAdapters1 function from the vTable
                EnumAdapters1 = factory.contents.lpVtbl.contents.EnumAdapters1
                EnumAdapters1_func = ctypes.WINFUNCTYPE(
                    comtypes.HRESULT,  # Return type
                    ctypes.POINTER(IDXGIFactory1),  # this pointer
                    ctypes.c_uint,  # Adapter index
                    ctypes.POINTER(ctypes.POINTER(IDXGIAdapter1))  # Adapter pointer
                )(EnumAdapters1)
                
                # Call EnumAdapters1
                hr = EnumAdapters1_func(factory, i, ctypes.byref(adapter))
                
                # DXGI_ERROR_NOT_FOUND (0x887A0002) means we've enumerated all adapters
                if hr == 0x887A0002:
                    break
                
                if hr != 0:  # Non-zero (except NOT_FOUND) is an error
                    print_result("warn", f"EnumAdapters1 returned error: {hr:#x}")
                    break
                
                # If we got here, we have a valid adapter
                adapters_found += 1
                
                # Try to get adapter description
                try:
                    desc = DXGI_ADAPTER_DESC1()
                    # Extract GetDesc1 function from the vTable
                    GetDesc1 = adapter.contents.lpVtbl.contents.GetDesc1
                    GetDesc1_func = ctypes.WINFUNCTYPE(
                        comtypes.HRESULT,  # Return type
                        ctypes.POINTER(IDXGIAdapter1),  # this pointer
                        ctypes.POINTER(DXGI_ADAPTER_DESC1)  # Description pointer
                    )(GetDesc1)
                    
                    # Call GetDesc1
                    hr = GetDesc1_func(adapter, ctypes.byref(desc))
                    
                    if hr == 0:  # S_OK
                        print(f"  Adapter {i}: {desc.Description}")
                        print(f"    VRAM: {desc.DedicatedVideoMemory / (1024*1024):.2f} MB")
                        print(f"    Vendor ID: {desc.VendorId}")
                    else:
                        print_result("warn", f"GetDesc1 returned error: {hr:#x}")
                except Exception as e:
                    print_result("warn", f"Error getting adapter description: {e}")
                
                # Release the adapter
                adapter.contents.lpVtbl.contents.Release()
                
                i += 1
            except Exception as e:
                print_result("warn", f"Error enumerating adapter {i}: {e}")
                break
        
        # Make sure to release the factory
        factory.contents.lpVtbl.contents.Release()
        
        if adapters_found > 0:
            print_result("pass", f"Found {adapters_found} graphics adapters")
        else:
            print_result("fail", "No graphics adapters found")
        
        return adapters_found > 0
    except Exception as e:
        print_result("fail", f"Error enumerating adapters: {e}")
        traceback.print_exc()
        return False

def test_d3d11_device_creation():
    """Test D3D11 device creation directly"""
    print_header("D3D11 DEVICE CREATION")
    
    # Define minimal structures and constants for D3D11 device creation
    D3D_DRIVER_TYPE_UNKNOWN = 0
    D3D_DRIVER_TYPE_HARDWARE = 1
    D3D_DRIVER_TYPE_WARP = 5
    
    D3D_FEATURE_LEVEL_11_1 = 0xb100
    D3D_FEATURE_LEVEL_11_0 = 0xb000
    D3D_FEATURE_LEVEL_10_1 = 0xa100
    D3D_FEATURE_LEVEL_10_0 = 0xa000
    
    D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
    D3D11_SDK_VERSION = 7
    
    class ID3D11Device_vTable(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            # Rest of the vTable...
        ]
    
    class ID3D11Device(ctypes.Structure):
        _fields_ = [("lpVtbl", ctypes.POINTER(ID3D11Device_vTable))]
    
    class ID3D11DeviceContext_vTable(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            # Rest of the vTable...
        ]
    
    class ID3D11DeviceContext(ctypes.Structure):
        _fields_ = [("lpVtbl", ctypes.POINTER(ID3D11DeviceContext_vTable))]
    
    # Try to create D3D11 device
    try:
        # Make sure d3d11.dll is available
        d3d11_dll = ctypes.windll.d3d11
        
        # Get D3D11CreateDevice function
        D3D11CreateDevice = d3d11_dll.D3D11CreateDevice
        D3D11CreateDevice.restype = comtypes.HRESULT
        D3D11CreateDevice.argtypes = [
            ctypes.c_void_p,  # pAdapter
            ctypes.c_uint,    # DriverType
            ctypes.c_void_p,  # Software
            ctypes.c_uint,    # Flags
            ctypes.POINTER(ctypes.c_uint),  # pFeatureLevels
            ctypes.c_uint,    # FeatureLevels
            ctypes.c_uint,    # SDKVersion
            ctypes.POINTER(ctypes.POINTER(ID3D11Device)),  # ppDevice
            ctypes.POINTER(ctypes.c_uint),  # pFeatureLevel
            ctypes.POINTER(ctypes.POINTER(ID3D11DeviceContext))  # ppImmediateContext
        ]
        
        # Feature levels to try
        feature_levels = [
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        ]
        feature_levels_array = (ctypes.c_uint * len(feature_levels))(*feature_levels)
        
        # Create device parameters
        p_device = ctypes.POINTER(ID3D11Device)()
        feature_level = ctypes.c_uint(0)
        p_context = ctypes.POINTER(ID3D11DeviceContext)()
        
        # Try hardware driver first
        print("Trying hardware D3D11 device creation...")
        hr = D3D11CreateDevice(
            None,  # No adapter specified
            D3D_DRIVER_TYPE_HARDWARE,
            None,  # Software
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            feature_levels_array,
            len(feature_levels),
            D3D11_SDK_VERSION,
            ctypes.byref(p_device),
            ctypes.byref(feature_level),
            ctypes.byref(p_context)
        )
        
        # If hardware fails, try WARP
        if hr != 0:
            print_result("warn", f"Hardware D3D11 device creation failed: {hr:#x}")
            print("Trying WARP (software) D3D11 device creation...")
            
            # Reset pointers
            p_device = ctypes.POINTER(ID3D11Device)()
            feature_level = ctypes.c_uint(0)
            p_context = ctypes.POINTER(ID3D11DeviceContext)()
            
            hr = D3D11CreateDevice(
                None,  # No adapter specified
                D3D_DRIVER_TYPE_WARP,
                None,  # Software
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                feature_levels_array,
                len(feature_levels),
                D3D11_SDK_VERSION,
                ctypes.byref(p_device),
                ctypes.byref(feature_level),
                ctypes.byref(p_context)
            )
        
        if hr == 0 and p_device and p_context:
            # Success!
            feature_level_major = (feature_level.value >> 12) & 0xF
            feature_level_minor = (feature_level.value >> 8) & 0xF
            print_result("pass", f"Successfully created D3D11 device with feature level {feature_level_major}.{feature_level_minor}")
            
            # Release resources
            p_device.contents.lpVtbl.contents.Release()
            p_context.contents.lpVtbl.contents.Release()
            return True
        else:
            print_result("fail", f"Failed to create D3D11 device: {hr:#x}")
            return False
            
    except Exception as e:
        print_result("fail", f"Error in D3D11 device creation: {e}")
        traceback.print_exc()
        return False

def test_desktop_duplication():
    """Test Desktop Duplication API manually"""
    print_header("DESKTOP DUPLICATION API")
    
    try:
        # First create a DXGI factory
        create_factory = ctypes.windll.dxgi.CreateDXGIFactory1
        factory_ptr = ctypes.c_void_p(0)
        
        dxgi_factory1_iid = "{770aae78-f26f-4dba-a829-253c83d1b387}"
        hr = create_factory(
            ctypes.byref(comtypes.GUID(dxgi_factory1_iid)),
            ctypes.byref(factory_ptr)
        )
        
        if hr != 0 or not factory_ptr.value:
            print_result("fail", f"Failed to create DXGI factory: {hr:#x}")
            return False
            
        print_result("pass", "Created DXGI factory")
        
        # Unfortunately, testing the full Desktop Duplication API requires a lot more
        # COM interface code than we can include here. We'll report that the factory
        # creation worked, which is a good sign.
        
        # For a complete test, we'd need to:
        # 1. Enumerate adapters with EnumAdapters1
        # 2. Enumerate outputs with EnumOutputs
        # 3. Create a D3D11 device
        # 4. Call IDXGIOutput1::DuplicateOutput
        # 5. Test frame acquisition
        
        print_result("info", "Full Desktop Duplication API test requires complex COM interaction.")
        print_result("info", "If DXGI factory creation succeeded, basic DirectX initialization is working.")
        print_result("info", "For detailed testing of the Desktop Duplication API, use the RapidShot diagnostic script.")
        
        return True
    except Exception as e:
        print_result("fail", f"Error testing Desktop Duplication API: {e}")
        return False

def test_rapidshot_imports():
    """Test importing RapidShot to identify import issues"""
    print_header("RAPIDSHOT IMPORT TEST")
    
    try:
        print("Attempting to import rapidshot...")
        import rapidshot
        print_result("pass", f"Successfully imported rapidshot from {rapidshot.__file__}")
        print_result("pass", f"RapidShot version: {getattr(rapidshot, '__version__', 'Unknown')}")
        
        # Check if key attributes and functions exist
        missing_attributes = []
        for attr in ["create", "device_info", "output_info", "clean_up", "ScreenCapture"]:
            if not hasattr(rapidshot, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            print_result("fail", f"Missing attributes in rapidshot: {', '.join(missing_attributes)}")
        else:
            print_result("pass", "All required functions found in rapidshot module")
            
        # Try importing core modules
        try:
            import rapidshot.core
            print_result("pass", "Successfully imported rapidshot.core")
            
            core_modules = ["Device", "Output", "Duplicator", "StageSurface"]
            missing_core = []
            for module in core_modules:
                if not hasattr(rapidshot.core, module):
                    missing_core.append(module)
            
            if missing_core:
                print_result("fail", f"Missing core modules: {', '.join(missing_core)}")
            else:
                print_result("pass", "All core modules found")
                
        except ImportError as e:
            print_result("fail", f"Failed to import rapidshot.core: {e}")
        
        # Try the DirectX API imports
        try:
            from rapidshot._libs import d3d11, dxgi
            print_result("pass", "Successfully imported DirectX API modules")
            
            # Check if key DirectX constants are defined
            if not hasattr(d3d11, "D3D_FEATURE_LEVEL_11_0"):
                print_result("warn", "D3D_FEATURE_LEVEL_11_0 not found in d3d11 module")
            if not hasattr(dxgi, "IDXGIFactory1"):
                print_result("warn", "IDXGIFactory1 not found in dxgi module")
                
        except ImportError as e:
            print_result("fail", f"Failed to import DirectX API modules: {e}")
        
        return True
    except ImportError as e:
        print_result("fail", f"Failed to import rapidshot: {e}")
        return False
    except Exception as e:
        print_result("fail", f"Unexpected error during rapidshot import: {e}")
        traceback.print_exc()
        return False

def test_rapidshot_device_info():
    """Test rapidshot.device_info() function to get device information"""
    print_header("RAPIDSHOT DEVICE INFO")
    
    try:
        import rapidshot
        
        # Try to get device info
        try:
            device_info = rapidshot.device_info()
            print(f"{Colors.CYAN}Device info:{Colors.RESET}\n{device_info}")
            
            if not device_info or device_info.strip() == "":
                print_result("warn", "device_info() returned empty result")
            else:
                print_result("pass", "Successfully retrieved device info")
                
            return True
        except Exception as e:
            print_result("fail", f"Error in device_info(): {e}")
            traceback.print_exc()
            return False
            
    except ImportError:
        print_result("fail", "Cannot import rapidshot module")
        return False

def test_rapidshot_output_info():
    """Test rapidshot.output_info() function to get output information"""
    print_header("RAPIDSHOT OUTPUT INFO")
    
    try:
        import rapidshot
        
        # Try to get output info
        try:
            output_info = rapidshot.output_info()
            print(f"{Colors.CYAN}Output info:{Colors.RESET}\n{output_info}")
            
            if not output_info or output_info.strip() == "":
                print_result("warn", "output_info() returned empty result")
            else:
                print_result("pass", "Successfully retrieved output info")
                
            return True
        except Exception as e:
            print_result("fail", f"Error in output_info(): {e}")
            traceback.print_exc()
            return False
            
    except ImportError:
        print_result("fail", "Cannot import rapidshot module")
        return False

def test_rapidshot_create():
    """Test rapidshot.create() function to create a ScreenCapture instance"""
    print_header("RAPIDSHOT CREATE")
    
    try:
        import rapidshot
        
        # Try to create a capture instance
        try:
            print("Attempting to create a capture instance...")
            screen = rapidshot.create(output_color="BGR")
            
            if screen is not None:
                print_result("pass", "Successfully created capture instance")
                
                # Try to grab a frame
                try:
                    print("Attempting to grab a frame...")
                    frame = screen.grab()
                    
                    if frame is not None:
                        height, width, channels = frame.shape
                        print_result("pass", f"Successfully grabbed frame: {width}x{height}x{channels}")
                    else:
                        print_result("warn", "grab() returned None. This could be normal if no screen updates occurred.")
                        
                    # Try again with a slight delay
                    import time
                    time.sleep(0.1)
                    
                    print("Attempting to grab a frame after delay...")
                    frame = screen.grab()
                    if frame is not None:
                        print_result("pass", "Successfully grabbed frame after delay")
                    else:
                        print_result("warn", "grab() still returned None after delay")
                    
                except Exception as e:
                    print_result("fail", f"Error grabbing frame: {e}")
                    traceback.print_exc()
                
                # Release resources
                try:
                    screen.release()
                    print_result("pass", "Successfully released capture resources")
                except Exception as e:
                    print_result("warn", f"Error releasing resources: {e}")
                
                # Clean up library
                try:
                    rapidshot.clean_up()
                    print_result("pass", "Successfully cleaned up library resources")
                except Exception as e:
                    print_result("warn", f"Error in clean_up(): {e}")
                
                return True
                
            else:
                print_result("fail", "create() returned None")
                return False
                
        except Exception as e:
            print_result("fail", f"Error in create(): {e}")
            traceback.print_exc()
            return False
            
    except ImportError:
        print_result("fail", "Cannot import rapidshot module")
        return False

def check_dependencies():
    """Check for required dependencies"""
    print_header("DEPENDENCY VERIFICATION")
    
    dependencies = [
        ("numpy", "1.19.0"),
        ("comtypes", "1.1.0"),
        ("opencv-python", "4.5.0", True),  # Optional
        ("pillow", "8.0.0", True),  # Optional
        ("cupy", "11.0.0", True)  # Optional
    ]
    
    all_required_dependencies = True
    
    for dep in dependencies:
        name = dep[0]
        min_version = dep[1]
        optional = len(dep) > 2 and dep[2]
        
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "Unknown")
            
            if "numpy" in name:
                # Special case for NumPy since it's critical
                if version < min_version:
                    print_result("warn", f"{name} version {version} is installed (min: {min_version})")
                    if not optional:
                        all_required_dependencies = False
                else:
                    print_result("pass", f"{name} version {version} is installed")
            else:
                print_result("pass", f"{name} version {version} is installed")
                
        except ImportError:
            if optional:
                print_result("warn", f"Optional dependency {name} is not installed")
            else:
                print_result("fail", f"Required dependency {name} is not installed")
                all_required_dependencies = False
    
    return all_required_dependencies

def main():
    """Main function to run all diagnostic tests"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}===== RAPIDSHOT DIRECTX DIAGNOSTIC TOOL ====={Colors.RESET}")
    print(f"Testing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    print_header("STARTING DIAGNOSTICS")
    
    # Basic system info
    test_system_info()
    
    # Dependencies
    check_dependencies()
    
    # DirectX DLLs
    test_directx_dlls()
    
    # GPU info
    test_gpu_info()
    
    # DXGI factory creation
    test_dxgi_factory_creation()
    
    # D3D11 device creation
    test_d3d11_device_creation()
    
    # Desktop Duplication API
    test_desktop_duplication()
    
    # RapidShot imports
    test_rapidshot_imports()
    
    # RapidShot device info
    test_rapidshot_device_info()
    
    # RapidShot output info
    test_rapidshot_output_info()
    
    # RapidShot create
    test_rapidshot_create()
    
    # Print summary
    print_summary()
    
    # Save results to file
    with open("rapidshot_diagnostics_results.txt", "w") as f:
        f.write(f"RapidShot DirectX Diagnostic Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"PASSED: {len(results['pass'])}\n")
        f.write(f"WARNINGS: {len(results['warn'])}\n")
        f.write(f"FAILED: {len(results['fail'])}\n\n")
        
        if results["fail"]:
            f.write("FAILED TESTS:\n")
            for fail in results["fail"]:
                f.write(f"  • {fail}\n")
            f.write("\n")
        
        if results["warn"]:
            f.write("WARNINGS:\n")
            for warn in results["warn"]:
                f.write(f"  • {warn}\n")
                
    print(f"\nDiagnostic results saved to {Colors.CYAN}rapidshot_diagnostics_results.txt{Colors.RESET}")

if __name__ == "__main__":
    main()
