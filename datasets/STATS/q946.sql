select  count(*) from postHistory as ph,          votes as v,  		users as u,  		badges as b  where u.Id = ph.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND u.Views<=270  AND u.DownVotes<=27  AND u.UpVotes>=0  AND u.UpVotes<=1160;