/*---------------------------------------------------------------------------
 *
 *    ExaDiS
 *
 *    Nicolas Bertin
 *    bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#ifndef EXADIS_VEC_H
#define EXADIS_VEC_H

#include <Kokkos_Core.hpp>

namespace ExaDiS {

/*---------------------------------------------------------------------------
 *
 *    Struct:        Vec3_t
 *
 *-------------------------------------------------------------------------*/
template <typename T>
struct Vec3_t
{
    T x, y, z;
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t() {
        x = (T)0; y = (T)0; z = (T)0;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t(const T &a) {
        x = a; y = a; z = a;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t(const T &_x, const T &_y, const T &_z) {
        x = _x; y = _y; z = _z;
    }
    
    Vec3_t(T *p) {
        x = p[0]; y = p[1]; z = p[2];
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    T operator[](const int i) const {
        if      (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else             return (T)0;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    T& operator[](const int i) {
        if      (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else             return x;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    bool operator==(const Vec3_t &a) {
        return ((x == a.x) && (y == a.y) && (z == a.z));
    }
    KOKKOS_FORCEINLINE_FUNCTION
    bool operator!=(const Vec3_t &a) {
        return !(*this == a);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t operator+=(const Vec3_t &a) {
        x += a.x; y += a.y; z += a.z;
        return *this;
    }
    KOKKOS_FORCEINLINE_FUNCTION
    void operator+=(const volatile Vec3_t &a) volatile {
        x += a.x; y += a.y; z += a.z;
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t operator-=(const Vec3_t &a) {
        x -= a.x; y -= a.y; z -= a.z;
        return *this;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    double norm2() const {
        return x*x + y*y + z*z;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    double norm() const {
        return sqrt(norm2());
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t normalized() const {
        double mag = norm();
        if (mag < 1e-20) return (*this);
        return Vec3_t<T>(x/mag, y/mag, z/mag);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t orthogonalize(const Vec3_t& a) const {
        double a2 = a.norm2();
        if (a2 <= 0.0) return (*this);
        return (*this) - dot((*this), a) / a2 * a;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t zero() {
        x = (T)0; y = (T)0; z = (T)0;
        return (*this);
    }
    
    void print() const {
        printf("%f %f %f\n", x, y, z);
    }
    void print(std::string s) const {
        printf("%s = %f %f %f\n", s.c_str(), x, y, z);
    }
};

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
Vec3_t<T> operator+(const Vec3_t<T> &a, const Vec3_t<T> &b) {
    return Vec3_t<T>(a.x+b.x, a.y+b.y, a.z+b.z);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
Vec3_t<T> operator-(const Vec3_t<T> &a, const Vec3_t<T> &b) {
    return Vec3_t<T>(a.x-b.x, a.y-b.y, a.z-b.z);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
Vec3_t<T> operator*(const int &s, const Vec3_t<T> &a) {
    return Vec3_t<T>(s * a.x, s * a.y, s * a.z);
}

template<typename T, typename U>
KOKKOS_FORCEINLINE_FUNCTION
Vec3_t<T> operator*(const U &s, const Vec3_t<T> &a) {
    return Vec3_t<T>(s * a.x, s * a.y, s * a.z);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T dot(const Vec3_t<T> &a, const Vec3_t<T> &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
Vec3_t<T> cross(const Vec3_t<T> &a, const Vec3_t<T> &b) {
    return Vec3_t<T>(a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
bool is_collinear(const Vec3_t<T> &v1, const Vec3_t<T> &v2, double tol=1e-5)
{
    double v1v2 = dot(v1, v2) / v1.norm() / v2.norm();
    return (fabs(fabs(v1v2)-1.0) < tol);
}

/*---------------------------------------------------------------------------
 *
 *    Class:    Mat33_t
 *
 *-------------------------------------------------------------------------*/
template<typename T>
class Mat33_t {
public:
    Vec3_t<T> rowx, rowy, rowz;
    
    KOKKOS_FORCEINLINE_FUNCTION T xx() const { return rowx.x; }
    KOKKOS_FORCEINLINE_FUNCTION T xy() const { return rowx.y; }
    KOKKOS_FORCEINLINE_FUNCTION T xz() const { return rowx.z; }
    KOKKOS_FORCEINLINE_FUNCTION T yx() const { return rowy.x; }
    KOKKOS_FORCEINLINE_FUNCTION T yy() const { return rowy.y; }
    KOKKOS_FORCEINLINE_FUNCTION T yz() const { return rowy.z; }
    KOKKOS_FORCEINLINE_FUNCTION T zx() const { return rowz.x; }
    KOKKOS_FORCEINLINE_FUNCTION T zy() const { return rowz.y; }
    KOKKOS_FORCEINLINE_FUNCTION T zz() const { return rowz.z; }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t<T> operator[](const int i) const {
        if      (i == 0) return rowx;
        else if (i == 1) return rowy;
        else if (i == 2) return rowz;
        else             return Vec3_t<T>((T)0);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Vec3_t<T>& operator[](const int i) {
        if      (i == 0) return rowx;
        else if (i == 1) return rowy;
        else if (i == 2) return rowz;
        else             return rowx;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION Vec3_t<T> colx() const { return Vec3_t<T>(xx(), yx(), zx()); }
    KOKKOS_FORCEINLINE_FUNCTION Vec3_t<T> coly() const { return Vec3_t<T>(xy(), yy(), zy()); }
    KOKKOS_FORCEINLINE_FUNCTION Vec3_t<T> colz() const { return Vec3_t<T>(xz(), yz(), zz()); }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t() { 
        rowx = Vec3_t<T>();
        rowy = Vec3_t<T>();
        rowz = Vec3_t<T>();
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t(const T &_v) { 
        rowx = Vec3_t<T>(_v);
        rowy = Vec3_t<T>(_v);
        rowz = Vec3_t<T>(_v);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t(const Vec3_t<T> &_x, const Vec3_t<T> &_y, const Vec3_t<T> &_z) { 
        rowx = _x;
        rowy = _y;
        rowz = _z;
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t(T *p) {
        rowx = Vec3_t<T>(&p[0]);
        rowy = Vec3_t<T>(&p[3]);
        rowz = Vec3_t<T>(&p[6]);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    bool operator==(const Mat33_t &b) {
        return ((rowx == b.rowx) && (rowy == b.rowy) && (rowz == b.rowz));
    }
    KOKKOS_FORCEINLINE_FUNCTION
    bool operator!=(const Mat33_t &b) {
        return !(*this == b);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t& operator+=(const Mat33_t &b) {
        rowx += b.rowx;
        rowy += b.rowy;
        rowz += b.rowz;
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t& operator-=(const Mat33_t &b) {
        rowx -= b.rowx;
        rowy -= b.rowy;
        rowz -= b.rowz;
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t set(
        const T &_xx, const T &_xy, const T &_xz,
        const T &_yx, const T &_yy, const T &_yz,
        const T &_zx, const T &_zy, const T &_zz) {
        rowx = Vec3_t<T>(_xx, _xy, _xz);
        rowy = Vec3_t<T>(_yx, _yy, _yz);
        rowz = Vec3_t<T>(_zx, _zy, _zz);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t set_rows(const Vec3_t<T> &_x, const Vec3_t<T> &_y, const Vec3_t<T> &_z) {
        rowx = _x;
        rowy = _y;
        rowz = _z;
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t set_columns(const Vec3_t<T> &_x, const Vec3_t<T> &_y, const Vec3_t<T> &_z) {
        (*this)[0][0] = _x.x; (*this)[1][0] = _x.y; (*this)[2][0] = _x.z;
        (*this)[0][1] = _y.x; (*this)[1][1] = _y.y; (*this)[2][1] = _y.z;
        (*this)[0][2] = _z.x; (*this)[1][2] = _z.y; (*this)[2][2] = _z.z;
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t zero() {
        rowx = Vec3_t<T>((T)0);
        rowy = Vec3_t<T>((T)0);
        rowz = Vec3_t<T>((T)0);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t eye() {
        rowx = Vec3_t<T>((T)1, (T)0, (T)0);
        rowy = Vec3_t<T>((T)0, (T)1, (T)0);
        rowz = Vec3_t<T>((T)0, (T)0, (T)1);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t diag(const T &_x, const T &_y, const T &_z) {
        rowx = Vec3_t<T>(_x, (T)0, (T)0);
        rowy = Vec3_t<T>((T)0, _y, (T)0);
        rowz = Vec3_t<T>((T)0, (T)0, _z);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t diag(const Vec3_t<T> &_v) {
        rowx = Vec3_t<T>(_v[0], (T)0, (T)0);
        rowy = Vec3_t<T>((T)0, _v[1], (T)0);
        rowz = Vec3_t<T>((T)0, (T)0, _v[2]);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t symmetric(const T &_xx, const T &_yy, const T &_zz, 
                      const T &_xy, const T &_xz, const T &_yz) {
        rowx = Vec3_t<T>(_xx, _xy, _xz);
        rowy = Vec3_t<T>(_xy, _yy, _yz);
        rowz = Vec3_t<T>(_xz, _yz, _zz);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t symmetric(T* p) {
        rowx = Vec3_t<T>(p[0], p[3], p[4]);
        rowy = Vec3_t<T>(p[3], p[1], p[5]);
        rowz = Vec3_t<T>(p[4], p[5], p[2]);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t voigt(const T &_xx, const T &_yy, const T &_zz, 
                  const T &_yz, const T &_xz, const T &_xy) {
        rowx = Vec3_t<T>(_xx, _xy, _xz);
        rowy = Vec3_t<T>(_xy, _yy, _yz);
        rowz = Vec3_t<T>(_xz, _yz, _zz);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t voigt(T* p) {
        rowx = Vec3_t<T>(p[0], p[5], p[4]);
        rowy = Vec3_t<T>(p[5], p[1], p[3]);
        rowz = Vec3_t<T>(p[4], p[3], p[2]);
        return (*this);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t normalized() const {
        return Mat33_t<T>(rowx.normalized(), rowy.normalized(), rowz.normalized());
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t transpose() const {
        return Mat33_t<T>().set_columns(rowx, rowy, rowz);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t adj() const {
        Mat33_t A = Mat33_t<T>();
        A[0][0] = (*this)[1][1]*(*this)[2][2] - (*this)[1][2]*(*this)[2][1];
        A[1][1] = (*this)[2][2]*(*this)[0][0] - (*this)[2][0]*(*this)[0][2];
        A[2][2] = (*this)[0][0]*(*this)[1][1] - (*this)[0][1]*(*this)[1][0];
        A[0][1] = (*this)[1][2]*(*this)[2][0] - (*this)[1][0]*(*this)[2][2];
        A[1][2] = (*this)[2][0]*(*this)[0][1] - (*this)[2][1]*(*this)[0][0];
        A[2][0] = (*this)[0][1]*(*this)[1][2] - (*this)[0][2]*(*this)[1][1];
        A[0][2] = (*this)[1][0]*(*this)[2][1] - (*this)[2][0]*(*this)[1][1];
        A[1][0] = (*this)[2][1]*(*this)[0][2] - (*this)[0][1]*(*this)[2][2];
        A[2][1] = (*this)[0][2]*(*this)[1][0] - (*this)[1][2]*(*this)[0][0];
        return A;
    }
    KOKKOS_FORCEINLINE_FUNCTION
    double det() const {
        Mat33_t A = adj();
        return ((*this)[0][0]*A[0][0] + (*this)[0][1]*A[0][1] + (*this)[0][2]*A[0][2]);
    }
    KOKKOS_FORCEINLINE_FUNCTION
    Mat33_t inverse() const {
        Mat33_t A = adj();
        double det = (*this)[0][0]*A[0][0] + (*this)[0][1]*A[0][1] + (*this)[0][2]*A[0][2];
        if (fabs(det) < 1e-20) return Mat33_t<T>((T)0);
        return (1.0/det)*A.transpose();
    }
    KOKKOS_FORCEINLINE_FUNCTION
    double trace() const {
        return (xx() + yy() + zz());
    }
    
    void print() const {
        printf("%f %f %f\n%f %f %f\n%f %f %f\n", 
        xx(), xy(), xz(), yx(), yy(), yz(), zx(), zy(), zz());
    }
    void print(std::string s) const {
        printf("%s =\n", s.c_str());
        this->print();
    }
};

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
Mat33_t<T> operator+(const Mat33_t<T> &a, const Mat33_t<T> &b) {
    return Mat33_t<T>(a.rowx + b.rowx, a.rowy + b.rowy, a.rowz + b.rowz);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
Mat33_t<T> operator-(const Mat33_t<T> &a, const Mat33_t<T> &b) {
    return Mat33_t<T>(a.rowx - b.rowx, a.rowy - b.rowy, a.rowz - b.rowz);
}

template<typename T, typename U>
KOKKOS_FORCEINLINE_FUNCTION
Mat33_t<T> operator*(const U &s, const Mat33_t<T> &a) {
    return Mat33_t<T>(s * a.rowx, s * a.rowy, s * a.rowz);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
Vec3_t<T> operator*(const Mat33_t<T> &M, const Vec3_t<T> &a) {
    return Vec3_t<T>(M[0][0] * a[0] + M[0][1] * a[1] + M[0][2] * a[2],
                     M[1][0] * a[0] + M[1][1] * a[1] + M[1][2] * a[2],
                     M[2][0] * a[0] + M[2][1] * a[1] + M[2][2] * a[2]);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
Mat33_t<T> operator*(const Mat33_t<T> &A, const Mat33_t<T> &B) {
    return Mat33_t<T>().set_columns(A*B.colx(), A*B.coly(), A*B.colz());
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
T dot(const Mat33_t<T> &A, const Mat33_t<T> &B) {
    return (A.xx()*B.xx() + A.xy()*B.xy() + A.xz()*B.xz() +
            A.yx()*B.yx() + A.yy()*B.yy() + A.yz()*B.yz() +
            A.zx()*B.zx() + A.zy()*B.zy() + A.zz()*B.zz());
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
Mat33_t<T> outer(const Vec3_t<T> &a, const Vec3_t<T> &b) {
    return Mat33_t<T>().set(a.x*b.x, a.x*b.y, a.x*b.z,
                            a.y*b.x, a.y*b.y, a.y*b.z,
                            a.z*b.x, a.z*b.y, a.z*b.z);
}


typedef Vec3_t<double> Vec3d;
typedef Vec3_t<float> Vec3f;
typedef Vec3_t<int> Vec3i;

typedef Mat33_t<double> Mat33d;
typedef Mat33_t<float> Mat33f;

#define Vec3 Vec3d
#define Mat33 Mat33d

} // namespace ExaDiS

#endif
