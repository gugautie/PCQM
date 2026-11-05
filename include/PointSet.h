#ifndef POINT_SET
#define POINT_SET


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>

#include <Eigen/Dense>
#include "tinyply.h"

class Point {
public:
  double x, y, z, nx, ny, nz, r, g, b, cie_L, cie_A, cie_B;

  friend std::ostream& operator<<(std::ostream& os, const Point& pts);
};

Point operator+(const Point& p1, const Point& p2) {
  Point p;
  p.x = p1.x + p2.x;
  p.y = p1.y + p2.y;
  p.z = p1.z + p2.z;
  return p;
}

Point operator+(const Point& p1, const Eigen::Vector3d& n) {
  Point p;
  p.x = p1.x + n(0);
  p.y = p1.y + n(1);
  p.z = p1.z + n(2);
  return p;
}

Point operator/(const Point& p1, const double a) {
  Point p;
  p.x = p1.x / a;
  p.y = p1.y / a;
  p.z = p1.z / a;
  return p;
}

Point operator*(const Point& p1, const double a) {
  Point p;
  p.x = p1.x * a;
  p.y = p1.y * a;
  p.z = p1.z * a;
  return p;
}

std::ostream& operator<<(std::ostream& os, const Point& pts) {
  os << "{" << pts.x << " , " << pts.y << " , " << pts.z << " }";
  return os;
}


Eigen::Matrix3d operator*(const Point& p1, const Point& p2) {
  Eigen::Matrix3d M(3, 3);
  M(0, 0) = p1.x * p2.x;
  M(0, 1) = p1.x * p2.y;
  M(0, 2) = p1.x * p2.z;
  M(1, 1) = p1.y * p2.y;
  M(1, 2) = p1.y * p2.z;
  M(2, 2) = p1.z * p2.z;
  M(1, 0) = M(0, 1);
  M(2, 0) = M(0, 2);
  M(2, 1) = M(1, 2);
  return M;
}

class PointSet {
public:
  std::vector<Point> pts;

  PointSet() { xmin = xmax = 0.0; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the distance between the vector "p1[0:size-1]" and the data point
  // with index "idx_p2" stored in the class:
  inline double kdtree_distance(const double* p1, const size_t idx_p2, size_t size) const {
    const double d0 = p1[0] - pts[idx_p2].x;
    const double d1 = p1[1] - pts[idx_p2].y;
    const double d2 = p1[2] - pts[idx_p2].z;
    return d0 * d0 + d1 * d1 + d2 * d2;
  }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the  "if/else's" are actually solved at compile time.
  inline double kdtree_get_pt(const size_t idx, int dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned
  // in "bb" so it can be avoided to redo it again.
  //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3
  // for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& bb) const {
    return false;
  }


  void readPointCloud(std::ifstream& f) {
    Point p;

    f >> p.x >> p.y >> p.z >> p.nx >> p.ny >> p.nz;

    unsigned int i = 0;
    xmin = xmax = p.x;
    ymin = ymax = p.y;
    zmin = zmax = p.z;

    while (f) {
      xmax = xmax > p.x ? xmax : p.x;
      ymax = ymax > p.y ? ymax : p.y;
      zmax = zmax > p.z ? zmax : p.z;
      xmin = xmin < p.x ? xmin : p.x;
      ymin = ymin < p.y ? ymin : p.y;
      zmin = zmin < p.z ? zmin : p.z;
      pts.push_back(p);
      ++i;
      f >> p.x >> p.y >> p.z >> p.nx >> p.ny >> p.nz;
    }
  }


  double xmin, ymin, zmin;
  double xmax, ymax, zmax;


  void loadPointCloud(std::vector<Point>& points) {
    pts.clear();
    std::vector<Point>::const_iterator pi = points.begin();
    unsigned int i = 0;
    xmin = xmax = pi->x;
    ymin = ymax = pi->y;
    zmin = zmax = pi->z;

    for (; pi != points.end(); ++pi) {
      xmax = xmax > pi->x ? xmax : pi->x;
      ymax = ymax > pi->y ? ymax : pi->y;
      zmax = zmax > pi->z ? zmax : pi->z;
      xmin = xmin < pi->x ? xmin : pi->x;
      ymin = ymin < pi->y ? ymin : pi->y;
      zmin = zmin < pi->z ? zmin : pi->z;
      pts.push_back(*pi);
      ++i;
    }
  }

  void addPoint(Point& p) {
    if (xmax - xmin < 1e-16) {
      xmin = xmax = p.x;
      ymin = ymax = p.y;
      zmin = zmax = p.z;
    } else {
      xmax = xmax > p.x ? xmax : p.x;
      ymax = ymax > p.y ? ymax : p.y;
      zmax = zmax > p.z ? zmax : p.z;
      xmin = xmin < p.x ? xmin : p.x;
      ymin = ymin < p.y ? ymin : p.y;
      zmin = zmin < p.z ? zmin : p.z;
    }
    pts.push_back(p);
  }

  int npts() const { return (int)pts.size(); }

  // TINIPLY READER
  inline std::vector<uint8_t> read_file_binary(const std::string& pathToFile) {
    std::ifstream file(pathToFile, std::ios::binary);
    std::vector<uint8_t> fileBufferBytes;

    if (file.is_open()) {
      file.seekg(0, std::ios::end);
      size_t sizeBytes = file.tellg();
      file.seekg(0, std::ios::beg);
      fileBufferBytes.resize(sizeBytes);
      if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
    } else {
      throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
	}  
    return fileBufferBytes;
  }

  struct memory_buffer : public std::streambuf {
    memory_buffer(char const* base, size_t size) {
      char* p(const_cast<char*>(base));
      this->setg(p, p, p + size);
    }
  };

  struct memory_stream : virtual memory_buffer, public std::istream {
    memory_stream(char const* base, size_t size)
        : memory_buffer(base, size), std::istream(static_cast<std::streambuf*>(this)) {}
  };

  struct float2 {
    float x, y;
  };
  struct float3 {
    float x, y, z;
  };
  struct double3 {
    double x, y, z;
  };

  struct unsigned_char3 {
    unsigned char r, g, b;
  };

  struct uint3 {
    uint32_t x, y, z;
  };
  struct uint4 {
    uint32_t x, y, z, w;
  };


  // void read_ply_file(const std::string& filepath, const bool preload_into_memory) {
  //   std::unique_ptr<std::istream> file_stream;
  //   std::vector<uint8_t> byte_buffer;

  //   try {
  //     // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a
  //     // stream is a net win for parsing speed, about 40% faster.
  //     if (preload_into_memory) {
  //       byte_buffer = read_file_binary(filepath);
  //       file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
  //     } else {
  //       file_stream.reset(new std::ifstream(filepath, std::ios::binary));
  //     }

  //     if (!file_stream || file_stream->fail()) throw std::runtime_error("failed to open " + filepath);

  //     tinyply::PlyFile file;
  //     file.parse_header(*file_stream);

  //     int point_size = 0;
  //     std::cout << "........................................................................\n";
  //     for (const auto& c : file.get_comments()) {
  //       std::cout << "Comment: " << c << std::endl;
  //     }
  //     for (const auto& e : file.get_elements()) {
  //       std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
  //       if (e.name == "vertex") {
  //         point_size = e.size;
  //         std::cout << "Pointset size set to " << point_size << std::endl;
  //       }
  //       for (const auto& p : e.properties) {
  //         std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")"
  //                   << std::endl;
  //       }
  //     }
  //     for (const auto& i : file.get_info())
	// 	std::cout << i << std::endl;
  //     std::cout << "........................................................................\n";

	  
  //     pts.clear();
  //     pts.resize(point_size);
  //     ///////////////////////////// MODIFICATION A PARTIR D'ICI /////////////////////////////

  //     // Tinyply treats parsed data as untyped byte buffers. See below for examples.
  //     std::shared_ptr<tinyply::PlyData> vertices, colors;

  //     // The header information can be used to programmatically extract properties on elements
  //     // known to exist in the header prior to reading the data. For brevity of this sample, properties
  //     // like vertex position are hard-coded:
  //     try {
  //       vertices = file.request_properties_from_element("vertex", {"x", "y", "z"},3);
  //     } catch (const std::exception& e) {
  //       std::cerr << "tinyply exception: " << e.what() << std::endl;
  //     }

  //     try {
  //       colors = file.request_properties_from_element("vertex", {"red", "green", "blue"},3);
  //     } catch (const std::exception& e) {
  //       std::cerr << "tinyply exception: " << e.what() << std::endl;
  //     }

  //     // read data from file
  //     file.read(*file_stream);

  //     std::vector< std::array<float, 3> > verts;
  //     std::vector< std::array<unsigned char, 3> > cols;
  //     // init boundaries
  //     xmin = ymin = zmin = std::numeric_limits<double>::max();
  //     xmax = ymax = zmax = std::numeric_limits<double>::min();

  //     // type casting to your own native types - Option A
  //     if (vertices) {
        
	// 	// const size_t numVerticesBytes = vertices->buffer.size_bytes();

  //       //std::cout << "bytes " << vertices->buffer.size_bytes() << std::endl;
  //       //std::cout << "count " << vertices->count << std::endl;
  //       //std::cout << "data size " << vertices->buffer.size_bytes() / vertices->count << std::endl;
  //       //std::cout << "expected data size " << sizeof(float3) << std::endl;
        
	// verts.resize(vertices->count);
	// if(tinyply::Type::FLOAT64 == file.get_elements()[0].properties[0].propertyType) {
  //         // Reading vertex using double or float64 //
  //         std::vector< std::array<double, 3> > tempDoubleVerts(vertices->count);
  //         std::memcpy(tempDoubleVerts.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
  //         for (size_t i = 0; i < tempDoubleVerts.size(); ++i) {
  //           for (size_t j = 0; j < 3; ++j) {
  //             verts[i][j] = static_cast<float>(tempDoubleVerts[i][j]); // Convert each double to float
  //           }
  //         } 
  //       } else {
  //         std::memcpy(verts.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
  //       }
  //       for (const auto& i : verts) {

	// 		// std::cout << i.x << " " << i.y << " " << i.z << std::endl;
  //       }
  //     }

  //     // type casting to your own native types - Option A
  //     if (colors) {
  //       const size_t numcolorsBytes = colors->buffer.size_bytes();
  //       cols.resize(colors->count);
  //       std::memcpy(cols.data(), colors->buffer.get(), numcolorsBytes);
  //     }

  //     Point p;
  //     for (int i_pts = 0; i_pts < point_size; i_pts++) {

  //       p.x = verts[i_pts][0];
  //       p.y = verts[i_pts][1];
  //       p.z = verts[i_pts][2];
  //       //std::cout << verts[i_pts].x << std::endl;
  //       if (colors) {
  //         p.r = cols[i_pts][0];
  //         p.g = cols[i_pts][1];
  //         p.b = cols[i_pts][2];
  //       } else {
  //         p.r = 1;
  //         p.g = 1;
  //         p.b = 1;
  //       }

  //       pts[i_pts] = p;


  //       xmax = xmax > p.x ? xmax : p.x;
  //       ymax = ymax > p.y ? ymax : p.y;
  //       zmax = zmax > p.z ? zmax : p.z;
  //       xmin = xmin < p.x ? xmin : p.x;
  //       ymin = ymin < p.y ? ymin : p.y;
  //       zmin = zmin < p.z ? zmin : p.z;
  //     }


  //   } catch (const std::exception& e) {
  //     std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  //   }
  // }

void read_ply_file(const std::string& filepath, const bool preload_into_memory) {
    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try {
      std::string format_line = "Unknown";

      if (preload_into_memory) {
        byte_buffer = read_file_binary(filepath);
        // Inspect header directly from byte_buffer without advancing the stream
        std::string header_preview(
            reinterpret_cast<const char*>(byte_buffer.data()),
            std::min<size_t>(byte_buffer.size(), 1024));
        size_t fmt_pos = header_preview.find("format ");
        if (fmt_pos != std::string::npos) {
          size_t end_pos = header_preview.find_first_of("\r\n", fmt_pos);
          format_line = header_preview.substr(fmt_pos, end_pos - fmt_pos);
        }
        file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
      } else {
        // Read format line with a separate temporary stream
        std::ifstream tmp_stream(filepath, std::ios::binary);
        std::string temp_line;
        while (std::getline(tmp_stream, temp_line)) {
          if (temp_line.find("format ") == 0) {
            format_line = temp_line;
            if (!format_line.empty() && format_line.back() == '\r') format_line.pop_back();
            break;
          }
          if (temp_line.find("end_header") == 0) break;
        }
        file_stream.reset(new std::ifstream(filepath, std::ios::binary));
      }

      if (!file_stream || file_stream->fail()) {
        throw std::runtime_error("failed to open " + filepath);
      }

      tinyply::PlyFile file;
      file.parse_header(*file_stream);

      int point_size = 0;
      std::cout << "========================================================================\n";
      std::cout << "[PLY DEBUG] Loading file: " << filepath << "\n";
      std::cout << "[PLY DEBUG] Header Format: " << format_line << "\n";

      if (format_line.find("binary_little_endian") != std::string::npos) {
        std::cout << "[PLY DEBUG] Endianness check: OK (Binary Little Endian)\n";
      } else if (format_line.find("ascii") != std::string::npos) {
        std::cout << "[PLY DEBUG] Endianness check: OK (ASCII format)\n";
      } else if (format_line.find("binary_big_endian") != std::string::npos) {
        std::cerr << "[PLY WARNING] File is Binary Big Endian! Verify tinyply endianness handling.\n";
      } else {
        std::cerr << "[PLY WARNING] Unrecognized PLY format specification!\n";
      }

      for (const auto& c : file.get_comments()) {
        std::cout << "[PLY DEBUG] Comment: " << c << std::endl;
      }
      for (const auto& e : file.get_elements()) {
        std::cout << "[PLY DEBUG] Element: " << e.name << " (" << e.size << ")" << std::endl;
        if (e.name == "vertex") {
          point_size = e.size;
        }
        for (const auto& p : e.properties) {
          std::cout << "\tProperty: " << p.name 
                    << " | Type: " << tinyply::PropertyTable[p.propertyType].str
                    << " | Stride: " << tinyply::PropertyTable[p.propertyType].stride << " bytes"
                    << std::endl;
        }
      }
      for (const auto& i : file.get_info()) {
        std::cout << "[PLY DEBUG] Info: " << i << std::endl;
      }

      if (point_size <= 0) {
        throw std::runtime_error("No vertices found in PLY header!");
      }

      pts.clear();
      pts.resize(point_size);

      std::shared_ptr<tinyply::PlyData> vertices, colors;

      try {
        vertices = file.request_properties_from_element("vertex", {"x", "y", "z"}, 3);
      } catch (const std::exception& e) {
        std::cerr << "[PLY ERROR] Missing vertex positions (x, y, z): " << e.what() << std::endl;
        throw;
      }

      try {
        colors = file.request_properties_from_element("vertex", {"red", "green", "blue"}, 3);
      } catch (const std::exception& e) {
        std::cerr << "[PLY DEBUG] Note: No (red, green, blue) properties found (" << e.what() << ")" << std::endl;
      }

      // Read data payload from file
      file.read(*file_stream);

      std::vector<std::array<float, 3>> verts(point_size);
      std::vector<std::array<unsigned char, 3>> cols;

      xmin = ymin = zmin = std::numeric_limits<double>::max();
      xmax = ymax = zmax = std::numeric_limits<double>::lowest();

      // Helper lambda to convert arbitrary vertex coordinate types to float
      auto copy_and_convert_vertices = [&](auto type_tag) {
        using T = decltype(type_tag);
        size_t expected_bytes = vertices->count * 3 * sizeof(T);
        if (vertices->buffer.size_bytes() != expected_bytes) {
          throw std::runtime_error("Vertex buffer byte size mismatch!");
        }
        const T* src = reinterpret_cast<const T*>(vertices->buffer.get());
        for (size_t i = 0; i < vertices->count; ++i) {
          verts[i][0] = static_cast<float>(src[i * 3 + 0]);
          verts[i][1] = static_cast<float>(src[i * 3 + 1]);
          verts[i][2] = static_cast<float>(src[i * 3 + 2]);
        }
      };

      if (vertices) {
        std::cout << "[PLY DEBUG] Vertex buffer: count = " << vertices->count
                  << ", total bytes = " << vertices->buffer.size_bytes()
                  << ", type = " << tinyply::PropertyTable[vertices->t].str << std::endl;

        switch (vertices->t) {
          case tinyply::Type::FLOAT32:
            std::cout << "[PLY DEBUG] Vertex type check: OK (FLOAT32)\n";
            if (vertices->buffer.size_bytes() != vertices->count * 3 * sizeof(float))
              throw std::runtime_error("Vertex FLOAT32 buffer size mismatch!");
            std::memcpy(verts.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
            break;
          case tinyply::Type::FLOAT64:
            std::cout << "[PLY DEBUG] Vertex type check: OK (FLOAT64 -> float)\n";
            copy_and_convert_vertices(double{});
            break;
          case tinyply::Type::UINT16:
            std::cout << "[PLY DEBUG] Vertex type check: OK (UINT16 / ushort -> float)\n";
            copy_and_convert_vertices(uint16_t{});
            break;
          case tinyply::Type::INT16:
            std::cout << "[PLY DEBUG] Vertex type check: OK (INT16 / short -> float)\n";
            copy_and_convert_vertices(int16_t{});
            break;
          case tinyply::Type::UINT32:
            std::cout << "[PLY DEBUG] Vertex type check: OK (UINT32 / uint -> float)\n";
            copy_and_convert_vertices(uint32_t{});
            break;
          case tinyply::Type::INT32:
            std::cout << "[PLY DEBUG] Vertex type check: OK (INT32 / int -> float)\n";
            copy_and_convert_vertices(int32_t{});
            break;
          case tinyply::Type::UINT8:
            std::cout << "[PLY DEBUG] Vertex type check: OK (UINT8 / uchar -> float)\n";
            copy_and_convert_vertices(uint8_t{});
            break;
          case tinyply::Type::INT8:
            std::cout << "[PLY DEBUG] Vertex type check: OK (INT8 / char -> float)\n";
            copy_and_convert_vertices(int8_t{});
            break;
          default:
            throw std::runtime_error("Unsupported vertex coordinate type: " +
                                     tinyply::PropertyTable[vertices->t].str);
        }
      }

      // Validate and copy colors
      if (colors) {
        std::cout << "[PLY DEBUG] Color buffer: count = " << colors->count
                  << ", total bytes = " << colors->buffer.size_bytes()
                  << ", type = " << tinyply::PropertyTable[colors->t].str << std::endl;

        if (colors->t == tinyply::Type::UINT8) {
          size_t expected_bytes = colors->count * 3 * sizeof(unsigned char);
          if (colors->buffer.size_bytes() != expected_bytes) {
            throw std::runtime_error("Color UINT8 buffer size mismatch!");
          }
          std::cout << "[PLY DEBUG] Color type check: OK (UINT8 / uchar)\n";
          cols.resize(colors->count);
          std::memcpy(cols.data(), colors->buffer.get(), colors->buffer.size_bytes());
        } else {
          std::cerr << "[PLY ERROR] Unsupported color type: "
                    << tinyply::PropertyTable[colors->t].str
                    << " (Expected uchar / uint8). Defaulting colors to (1,1,1)." << std::endl;
          colors.reset();
        }
      }

      Point p;
      for (int i_pts = 0; i_pts < point_size; i_pts++) {
        p.x = verts[i_pts][0];
        p.y = verts[i_pts][1];
        p.z = verts[i_pts][2];

        if (colors) {
          p.r = cols[i_pts][0];
          p.g = cols[i_pts][1];
          p.b = cols[i_pts][2];
        } else {
          p.r = 1;
          p.g = 1;
          p.b = 1;
        }

        pts[i_pts] = p;

        xmax = xmax > p.x ? xmax : p.x;
        ymax = ymax > p.y ? ymax : p.y;
        zmax = zmax > p.z ? zmax : p.z;
        xmin = xmin < p.x ? xmin : p.x;
        ymin = ymin < p.y ? ymin : p.y;
        zmin = zmin < p.z ? zmin : p.z;
      }

      if (point_size > 0) {
        std::cout << "[PLY DEBUG] Sample Point[0]: pos=("
                  << pts[0].x << ", " << pts[0].y << ", " << pts[0].z << "), rgb=("
                  << pts[0].r << ", " << pts[0].g << ", " << pts[0].b << ")\n";
        std::cout << "[PLY DEBUG] Bounding Box: X[" << xmin << ", " << xmax << "] "
                  << "Y[" << ymin << ", " << ymax << "] "
                  << "Z[" << zmin << ", " << zmax << "]\n";
      }
      std::cout << "========================================================================\n";

    } catch (const std::exception& e) {
      std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
      std::exit(EXIT_FAILURE); // Exit cleanly instead of segfaulting in KDTree build
    }
  }

};

#endif
