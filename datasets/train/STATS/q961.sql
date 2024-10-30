select  count(*) from votes as v,  		posts as p,          users as u where v.UserId = p.OwnerUserId 	and p.OwnerUserId = u.Id  AND p.CommentCount>=0  AND u.UpVotes>=0  AND u.UpVotes<=34;
