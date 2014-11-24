function switchingcontrol_par
% This function solves the parabolic switching control problem
%  min 1/2 \|y-yd\|^2 + alpha/2 \int_0^T |u|^2 + beta |u_1u_2|_0 dt
%  s.t. y_t-\Delta y = \chi_{\omega_1}(x)u_1(t) + \chi_{\omega_2}(x)u_2(t)
% using the approach described in the paper
%    "A convex analysis approach to optimal controls with 
%     switching structure for partial differential equations"
% by Christian Clason, Kazufumi Ito and Karl Kunisch, see
% http://math.uni-graz.at/mobis/publications/SFB-Report-2014-001.pdf.
%
% August 7, 2014             Christian Clason <christian.clason@uni-due.de>
%                                                     http://udue.de/clason

%% setup
Nh    = 128;     % number of spatial grid points
Nt    = 512;     % number of time steps
alpha = 1e-1;    % quadratic penalty
beta  = 1e0;     % switching penalty
maxit = 99;      % max iterations in semismooth Newton method
tmin  = 1e-12;   % min step size for line search
tol   = 1e-6;    % terminate Newton if residual norm smaller than tol

%  grid, differential operators
[L,Ms,It,xx,tt] = assembleFEM(Nh,Nt);
tplot = @(n,f,s) tplot_(n,f,s,Nh,Nt,xx,tt);
stack = @(x) x(:);

% control domains
om1 = xx(:,1) < -1/2;
om2 = xx(:,1) > 1/2;
dplot = @(n,f,s) dplot_(n,f,s,Nh,Nt,xx,tt,om1,om2);

% control operator
BT = [kron(It,om1)';kron(It,om2)']*Ms;    BB = BT';

% target
u0 = 63.0*(abs(-1+tt-xx)<0.1);
yd = L\(Ms*u0(:));
dplot(1,yd,'target');

%% initialization
y = zeros(Nh*Nt,1);
p = zeros(Nh*Nt,1);
As_old = zeros(Nt,7);
sqab = sqrt(2*alpha*beta);

% constant blocks of Newton matrix
Lyy = Ms;
Lpy = -L; Lyp = L';

% continuation strategy
for gamma = 10.^(-(0:16))
    fprintf('\nSolving for gamma = %1.0e\n', gamma);
    fprintf('It\tupdate\tresidual\tstep size\tQ1\tQ2\tQ0\tQ10\tQ20\tQ00\tQ12\n');
    
    % semismooth Newton iteration
    it = 1;    nold = 1e99;    tau = 1;
    while true
        % compute active sets
        q = BT*p; q1 = q(1:Nt); q2 = q(1+Nt:end);
        As = zeros(Nt,7);
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
        Ly = L'*p - Ms*(yd-y);
        Lp = -L*y + stack(BB*Hg);
        F = -[Ly; Lp];
        ng = norm(F);
        
        % line search
        if ng >= nold        % if no decrease: backtrack (never on first iteration)
            tau = tau/2;
            y = y - tau*dx(1:Nh*Nt);
            p = p - tau*dx(1+Nh*Nt:end);
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
            (alpha+gamma)/(gamma*(2*alpha+gamma))*As(:,7),0,Nt,Nt);
        H12 = spdiags(-alpha/(gamma*(2*alpha+gamma))*As(:,7).*sign(q1).*sign(q2),0,Nt,Nt);
        H22 = spdiags(1/(alpha+gamma)*(As(:,2)+As(:,3)+As(:,5)) + ...
            1/gamma*(As(:,4)+As(:,6)) + ...
            (alpha+gamma)/(gamma*(2*alpha+gamma))*As(:,7),0,Nt,Nt);
        Lpp = BB*[H11 H12; H12 H22]*BT;
        
        H = [Lyy Lyp; Lpy  Lpp];
        
        % solve for Newton direction
        dx = H\F;
        y  = y + dx(1:Nh*Nt);
        p  = p + dx(1+Nh*Nt:2*Nh*Nt);
        
    end % Newton
    
    % compute control from optimality condition, plot
    u = [Hg(1:Nt),Hg(1+Nt:end)];
    figure(2);plot(tt(1,:),u);   title('control')
    tplot(3,y,'state');
    
    % terminate continuation if Newton iteration converged in one step
    if it==1 && gamma < alpha
        break;
    end
    
end % continuation

end % function switchingcontrol_par

%%
function [L,M,It,xx,tt] = assembleFEM(Nh,Nt)
% grid
x = linspace(-1,1,Nh);
t = linspace(0,2,Nt);
[xx,tt] = ndgrid(x,t);

% setup finite element discretization
h   = x(2)-x(1);                   % mesh size
tau = t(2)-t(1);                   % time step size

% spatial discretization
ex = ones(Nh,1);
et = ones(Nt,1);
D2 = spdiags([-ex 2*ex -ex]/h,-1:1,Nh,Nh);    % A_h
Mx = spdiags([ex 4*ex ex]*(h/6),-1:1,Nh,Nh);  % M_h
Dt = spdiags([-et et]/tau,-1:0,Nt,Nt);
It = speye(Nt,Nt);

% space-time discretization
L = kron(Dt,Mx) + kron(It,D2);     % parabolic operator
M = kron(It,Mx);                   % mass matrix M_\sigma
end % function assembleFEM

function tplot_(n,f,s,Nh,Nt,xx,tt)
figure(n); surf(tt,xx,reshape(f,Nh,Nt)); axis tight
colormap summer
shading interp; lighting phong; camlight headlight; alpha(0.8);
title(s); xlabel('t'); ylabel('x');
drawnow;
end % tplot_ function

function dplot_(n,f,s,Nh,Nt,xx,tt,om1,om2)
figure(n); surf(tt,xx,reshape(f,Nh,Nt)); axis tight
colormap summer
shading interp; lighting phong; camlight headlight; alpha(0.8);
hold on
contour(tt,xx,reshape(1.0*(om1+om2)*ones(1,Nt),Nh,Nt),1,'LineWidth',2);
hold off
title(s); xlabel('t'); ylabel('x');
drawnow;
end % dplot_ function
