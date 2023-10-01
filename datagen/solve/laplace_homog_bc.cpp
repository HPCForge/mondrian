#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>

using namespace std;
using namespace mfem;

void solve_mesh(Mesh& mesh,
	              int order,
		      bool static_cond,
		      bool pa,
		      bool fa,
		      bool algebraic_ceed,
		      std::filesystem::path output_dir,
		      std::string path_stem);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_dir = "";
   const char *out_dir = "";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_dir, "-m", "--mesh",
                  "Mesh directory to use.");
   args.AddOption(&out_dir, "-d", "--outdir",
		  "Directory to save solutions.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   std::filesystem::path output_dir(out_dir);
   if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directory(output_dir);
   }

   for (auto& entry : std::filesystem::directory_iterator(mesh_dir)) {
      std::filesystem::path mesh_path = entry.path();
      std::string path_stem = mesh_path.stem().string();

      Mesh mesh(mesh_path.string(), 1, 1);
      solve_mesh(mesh, order, static_cond, pa, fa, algebraic_ceed, output_dir, path_stem);
   }

   return 0;
}

void solve_mesh(Mesh& mesh,
	        int order,
		bool static_cond,
		bool pa,
		bool fa,
		bool algebraic_ceed,
		std::filesystem::path output_dir,
		std::string path_stem) {
   int dim = mesh.Dimension();

   {
      int ref_levels =
         (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
   }
   else
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   a.RecoverFEMSolution(X, b, x);


   std::string refined_mesh_path = output_dir / (path_stem + "_refined.mesh");
   std::string sol_path = output_dir / (path_stem + "_sol.gf");

   std::cout << "writing mesh to " << refined_mesh_path << '\n';
   std::cout << "writing sol to " << sol_path << '\n';

   ofstream mesh_ofs(refined_mesh_path);
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs(sol_path);
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   if (delete_fec)
   {
      delete fec;
   }
}
