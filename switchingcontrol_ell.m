function switchingcontrol_ell
% This function solves the elliptic switching control problem
%  min 1/2 \|y-yd\|^2 + alpha/2 \int |u|^2 + beta |u_1u_2|_0 dx_1
%      s.t. -\Delta y = \chi_{\omega_1} u_1 + \chi_{\omega_2} u_2 
% using the approach described in the paper
%    "A convex analysis approach to optimal controls with 
%     switching structure for partial differential equations"
% by Christian Clason, Kazufumi Ito and Karl Kunisch, see
% http://math.uni-graz.at/mobis/publications/SFB-Report-2014-001.pdf.
%
% August 7, 2014             Christian Clason <christian.clason@uni-due.de>
%                                                     http://udue.de/clason

%% setup
N     = 128;     % number of spatial grid points
alpha = 1e-3;    % quadratic penalty
beta  = 1e-3;    % switching penalty
maxit = 99;      % max iterations in semismooth Newton method
tmin  = 1e-12;   % min step size for line search
tol   = 1e-6;    % terminate Newton if residual norm smaller than tol

%  grid, differential operators
[A,M,xx,yy] = assembleFEM(N);
tplot = @(n,f,s) tplot_(n,f,s,N,xx,yy);

% control domains
om1 = xx(1,:) < 1/4;
om2 = xx(1,:) > 3/4;
dplot = @(n,f,s) dplot_(n,f,s,N,xx,yy,om1,om2);

% control operator
BT = [kron(speye(N),om1')';kron(speye(N),om2')']*M';    BB = BT';

% target
yd = xx.*sin(2*pi*xx).*sin(2*pi*yy);
yd = yd(:);
dplot(1,yd,'target and control domains');

%% initialization
y = zeros(N*N,1);
p = zeros(N*N,1);
As_old = zeros(N,7);
sqab = sqrt(2*alpha*beta);

% constant blocks of Newton matrix
Lyy = M;
Lpy = A; Lyp = A';

% continuation strategy
for gamma = 10.^(-(0:16))
    fprintf('\nSolving for gamma = %1.0e\n', gamma);
    fprintf('It\tupdate\tresidual\tstep size\tQ1\tQ2\tQ0\tQ10\tQ20\tQ00\tQ12\n');

    % semismooth Newton iteration
    it = 1;    nold = 1e99;    tau = 1; 
    while true
        % compute active sets
        q = BT*p; q1 = q(1:N); q2 = q(1+N:end);
        As(:,1) = (abs(q1) > (1+gamma/alpha)*abs(q2)) & (abs(q2) < sqab);
        As(:,2) = (abs(q2) > (1+gamma/alpha)*abs(q1)) & (abs(q1) < sqab);
        As(:,3) = (abs(q1) > (1+gamma/alpha)*sqab) & (abs(q2) > (1+gamma/alpha)*sqab);
        As(:,4) = (sqab <= abs(q1)) & (abs(q1) <= (1+gamma/alpha)*sqab) & ...
            (abs(q2) > (1+gamma/alpha)*sqab);
        As(:,5) = (sqab <= abs(q2)) & (abs(q2) <= (1+gamma/alpha)*sqab) & ...
            (abs(q1) > (1+gamma/alpha)*sqab);
        As(:,6) = (sqab <= abs(q1)) & (abs(q1) <= (1+gamma/alpha)*sqab) & ...
            (sqab <= abs(q2)) & (abs(q2) <= (1+gamma/alpha)*sqab) & ...
            ((abs(q1)+abs(q2)) > (2+gamma/alpha)*sqab);
        As(:,7) = (alpha/(alpha+gamma)*abs(q2) <= abs(q1)) & ...
            (abs(q1) <= (1+gamma/alpha)*abs(q2)) & ...
            ((abs(q1)+abs(q2)) <= (2+gamma/alpha)*sqab);
        
        Hg = [1/(alpha+gamma)*(As(:,1)+As(:,3)+As(:,4)).*q1 + ...
            1/gamma*(As(:,5)+As(:,6)).*(q1-sign(q1)*sqab) + ...
            1/(gamma*(2*alpha+gamma))*As(:,7).*((alpha+gamma)*q1-alpha*sign(q1).*abs(q2)); ...
            1/(alpha+gamma)*(As(:,2)+As(:,3)+As(:,5)).*q2 + ...
            1/gamma*(As(:,4)+As(:,6)).*(q2-sign(q2)*sqab) + ...
            1/(gamma*(2*alpha+gamma))*As(:,7).*((alpha+gamma)*q2-alpha*sign(q2).*abs(q1))];
        
        % gradient
        Ly = A'*p + M*(y-yd);
        Lp = A*y - BB*Hg;
        F = -[Ly; Lp];
        ng = norm(F);
        
        % line search
        if ng >= nold        % if no decrease: backtrack (never on first iteration)
            tau = tau/2;
            y = y - tau*dx(1:N*N);
            p = p - tau*dx(1+N*N:end);
            if tau < tmin    % terminate Newton iteration
                disp('step size too small')
                break;       
            else             % bypass rest of while loop; compute new gradient
                continue; 
            end
        end
        % decrease (or on first iteration): accept step
        update = nnz(As-As_old);
        fprintf('%i\t%d\t\t%1.3e\t%1.3e\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n',...
            it,update,ng,tau,sum(As,1));
        
        % terminate Newton?
        if update == 0 && ng < tol  % success, solution found
            break;
        elseif it == maxit          % failure, too many iterations
            break;
        end

        % otherwise update information, continue
        it = it+1;   nold = ng;   tau = 1;   As_old = As;
        
        % Newton matrix: iteration dependent block
        H11 = spdiags(1/(alpha+gamma)*(As(:,1)+As(:,3)+As(:,4)) + ...
            1/gamma*(As(:,5)+As(:,6)) + ...
            (alpha+gamma)/(gamma*(2*alpha+gamma))*As(:,7),0,N,N);
        H12 = spdiags(-alpha/(gamma*(2*alpha+gamma))*As(:,7).*sign(q1).*sign(q2),0,N,N);
        H22 = spdiags(1/(alpha+gamma)*(As(:,2)+As(:,3)+As(:,5)) + ...
            1/gamma*(As(:,4)+As(:,6)) + ...
            (alpha+gamma)/(gamma*(2*alpha+gamma))*As(:,7),0,N,N);
        Lpp = -BB*[H11 H12; H12 H22]*BT;
        
        H = [Lyy Lyp; Lpy  Lpp];
        
        % solve for Newton direction
        dx = H\F;
        y  = y + dx(1:N*N);
        p  = p + dx(1+N*N:2*N*N);
        
    end % Newton
    
    % compute control from optimality condition, plot
    u = [Hg(1:N),Hg(1+N:end)];
    figure(2);plot(xx(1,:),u);   title('control')
    tplot(3,y,'state');
    
    % terminate continuation if Newton iteration converged in one step
    if it==1 && gamma < alpha
        break;
    end
    
end % continuation

end

%%
function [K,M,xx,yy] = assembleFEM(n)
a   = 0;    b = 1;       % computational domain [a,b]^2
nel = 2*(n-1)^2;         % number of nodes
h2  = ((b-a)/(n-1))^2;   % Jacobi determinant of transformation (2*area(T))

% nodes
[xx,yy] = meshgrid(linspace(0,1,n));

% triangulation
tri = zeros(nel,3);
ind = 1;
for i = 1:n-1
    for j = 1:n-1
        node         = (i-1)*n+j+1;              % two triangles at node
        tri(ind,:)   = [node node-1 node+n];     % triangle 1 (lower left)
        tri(ind+1,:) = [node+n-1 node+n node-1]; % triangle 2 (upper right)
        ind = ind+2;
    end
end

% Mass and stiffness matrices
Ke = 1/2 * [2 -1 -1 -1 1 0 -1 0 1]';   % elemental stiffness matrix
Me = h2/24 * [2 1 1 1 2 1 1 1 2]';     % elemental mass matrix

ent = 9*nel;
row = zeros(ent,1);
col = zeros(ent,1);
valk = zeros(ent,1);
valm = zeros(ent,1);

ind = 1;
for el=1:nel
    ll       = ind:(ind+8);            % local node indices
    gl       = tri(el,:);              % global node indices
    row(ll)  = gl([1;1;1],:); rg = gl';
    col(ll)  = rg(:,[1 1 1]);
    valk(ll) = Ke;
    valm(ll) = Me;
    ind      = ind+9;
end
M = sparse(row,col,valm);
K = sparse(row,col,valk);

% modify matrices for homogenenous Dirichlet conditions
bdnod = [find(abs(xx-a) < eps); find(abs(yy-a) < eps); ...
    find(abs(xx-b) < eps); find(abs(yy-b) < eps)];
M(bdnod,:) = 0;
K(bdnod,:) = 0;  K(:,bdnod) = 0;
for j = bdnod'
    K(j,j) = 1; %#ok<SPRIX>
end
end

%%
function tplot_(n,f,s,N,x,y)
figure(n); 
surf(x,y,reshape(f,N,N));
colormap summer
shading interp; lighting phong; camlight headlight; alpha(0.8);
title(s); xlabel('x_1'); ylabel('x_2');
drawnow;
end % tplot_ function

function dplot_(n,f,s,N,x,y,om1,om2)
figure(n); 
surf(x,y,reshape(f,N,N));
colormap summer
shading interp; lighting phong; camlight headlight; alpha(0.8);
hold on
contour(x,y,reshape(1.0*(om1+om2)'*ones(1,N),N,N),1,'LineWidth',2);
hold off
title(s); xlabel('x_1'); ylabel('x_2');
drawnow;
end % dplot_ function
